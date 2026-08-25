import json
import sys
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase
from tokenizers import normalizers
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # 上一级目录
sys.path.insert(0, parent_dir)  # 将父目录添加到Python路径

from configs.llm_utils import llm_data_dir, llm_model_dir

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 配置参数
VOCAB_SIZE = 6400
MIN_FREQUENCY = 10
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[EOS]", "[BOS]"]
TRAIN_DATA_PATH =llm_data_dir+ "/pretrain_t2t.jsonl"
TOKENIZER_SAVE_PATH = llm_model_dir+"/bpe_tokenizer.json"
import logging
# 批量处理参数
BATCH_SIZE = 600  # 每次处理多少行
MAX_LINES = None  # 限制处理行数，None表示全部处理，测试时可以设置小一点


def stream_jsonl(file_path, max_lines=None):
    """
    流式读取JSONL文件，每次yield一个文本
    使用生成器避免一次性加载整个文件
    """
    logger.info(f"开始流式读取文件: {file_path}")

    line_count = 0
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 如果设置了最大行数限制
                if max_lines is not None and line_count >= max_lines:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    line_count += 1
                    # 提取文本 - 根据你的数据格式调整
                    contents = [item.get('content') for item in data.get('conversations', []) if item.get('content')]
                    if contents:
                        yield "\n".join(contents)
                    # else:
                    #     print(data)

                    contents = [data.get('text')]
                    if contents:
                        yield "\n".join(contents)
                    # else:
                    #     print(data)

                except json.JSONDecodeError as e:
                    logger.warning(f"第{line_count}行JSON解析失败: {e}")
                    continue
                except Exception as e:
                    logger.warning(f"处理第{line_count}行时出错: {e}")
                    continue

    except FileNotFoundError:
        logger.error(f"文件不存在: {file_path}")
        raise

    logger.info(f"总共处理了 {line_count} 行数据")


def batch_generator(file_path, batch_size=10000, max_lines=None):
    """
    批量生成器，每次yield一个批次
    用于减少yield次数，提高训练效率
    """
    batch = []
    count = 0

    for text in stream_jsonl(file_path, max_lines):
        batch.append(text)
        count += 1

        if len(batch) >= batch_size:
            yield batch
            batch = []
            # 打印进度
            if count % (batch_size * 10) == 0:
                logger.info(f"已处理 {count} 条文本")

    # 处理最后一批
    if batch:
        yield batch


def train_bpe_tokenizer_streaming(file_path, vocab_size, special_tokens,
                                  min_frequency=5, batch_size=10000,
                                  max_lines=None):
    """
    流式训练BPE Tokenizer，不会将所有数据加载到内存
    """
    logger.info(f"词表大小: {vocab_size}")
    if max_lines:
        logger.info(f"限制处理行数: {max_lines}")

    # 1. 初始化BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # 2. 设置normalizer（规范化）
    #在tokenization之前对原始文本进行预处理，目的是统一文本格式，减少不必要的变体，提高tokenization的效果。
    #为什么需要Normalizer？
        # 在自然语言中，同一个字符可能有多种表示方式，这会导致：
        # 词表膨胀：相同意思的文本被拆分成不同的token
        # 语义损失：不同表示的相同字符被当作不同token处理
        # 数据稀疏：低频变体无法被有效学习

    tokenizer.normalizer = normalizers.Sequence([
        NFD(),#NFD是Unicode规范化的一种形式，全称是Normalization Form Decomposition（标准化分解形式）
        Lowercase(),  # 如果你需要保留大小写，可以注释掉这一行
    ])

    # 3. 设置pre-tokenizer（预分词）
    tokenizer.pre_tokenizer = Whitespace()

    # 4. 设置post-processor（后处理）
    #对生成的token序列进行后处理，主要目的是：
        # 添加特殊标记（如[CLS]、[SEP]）
        # 处理成对输入（如问答对、句子对）
        # 为模型准备标准格式（特别是BERT类模型）
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 2),
            ("[SEP]", 3),
        ],
    )

    # 5. 设置trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        show_progress=True,
        continuing_subword_prefix="##",
        end_of_word_suffix="</w>",
        limit_alphabet=200,
        max_token_length=16,
    )

    # 6. 使用批量生成器进行训练
    logger.info("开始训练（流式处理）...")

    # 创建批量生成器
    batches = batch_generator(file_path, batch_size, max_lines)

    # 训练 - tokenizers库会自动迭代generator
    # 注意：这里训练时tokenizers会多次遍历数据，但内部使用流式处理
    tokenizer.train_from_iterator(batches, trainer=trainer)

    logger.info(f"训练完成！词表大小: {tokenizer.get_vocab_size()}")

    return tokenizer


def train_with_progress(file_path, vocab_size, special_tokens,
                        min_frequency=5, batch_size=10000, max_lines=None):
    """
    带进度显示的流式训练
    """
    # 首先统计总行数（如果需要显示进度）
    total_lines = None
    if max_lines is None:
        # 快速统计行数（可选，会额外读取一次文件）
        # 对于10GB文件，这个操作可能很耗时，建议只在测试时使用
        # total_lines = sum(1 for _ in open(file_path, 'r', encoding='utf-8'))
        # logger.info(f"文件总行数: {total_lines}")
        pass

    # 开始训练
    tokenizer = train_bpe_tokenizer_streaming(
        file_path=file_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=min_frequency,
        batch_size=batch_size,
        max_lines=max_lines
    )

    return tokenizer


def analyze_vocabulary(tokenizer):
    """分析词表统计信息"""
    vocab = tokenizer.get_vocab()

    logger.info("=" * 50)
    logger.info("词表统计分析:")
    logger.info(f"总词表大小: {len(vocab)}")

    # 统计不同长度的token
    token_lengths = {}
    for token, idx in vocab.items():
        length = len(token)
        token_lengths[length] = token_lengths.get(length, 0) + 1

    logger.info("Token长度分布 (前10个):")
    for length, count in sorted(token_lengths.items())[:10]:
        logger.info(f"  长度 {length}: {count} 个token")

    # 显示一些示例token
    logger.info("\n词表示例 (前20个):")
    for i, (token, idx) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:20]):
        logger.info(f"  {idx}: {token}")

    return vocab


def test_tokenizer(tokenizer, test_texts=None):
    """测试Tokenizer效果"""
    logger.info("=" * 50)
    logger.info("开始测试Tokenizer...")

    if test_texts is None:
        test_texts = [
            "你好，这是一个测试文本。",
            "The quick brown fox jumps over the lazy dog.",
            "这是一个中英混合的测试：Hello World!",
            "让我们测试一些特殊符号：@#$%^&*()",
            "数字测试：1234567890",
        ]

    for i, text in enumerate(test_texts, 1):
        logger.info(f"\n测试样本 {i}:")
        logger.info(f"原始文本: {text}")

        # 编码
        encoded = tokenizer.encode(text)
        logger.info(f"Token数量: {len(encoded.tokens)}")
        logger.info(f"Tokens (前20个): {encoded.tokens[:20]}...")
        logger.info(f"Token IDs (前20个): {encoded.ids[:20]}...")

        # 解码
        decoded = tokenizer.decode(encoded.ids)
        logger.info(f"解码还原: {decoded}")
        logger.info("-" * 30)


def save_tokenizer(tokenizer, path):
    """保存Tokenizer"""
    logger.info(f"保存Tokenizer到: {path}")
    tokenizer.save(path)
    logger.info("保存成功！")


def main():
    """主函数"""
    try:
        # 训练参数
        config = {
            'file_path': TRAIN_DATA_PATH,
            'vocab_size': VOCAB_SIZE,
            'special_tokens': SPECIAL_TOKENS,
            'min_frequency': MIN_FREQUENCY,
            'batch_size': BATCH_SIZE,
            'max_lines': MAX_LINES,  # 设置为None处理全部，测试时可设置如100000
        }

        # 1. 训练Tokenizer（流式处理）
        tokenizer = train_with_progress(**config)

        # 2. 分析词表
        analyze_vocabulary(tokenizer)

        # 3. 测试Tokenizer
        test_tokenizer(tokenizer)

        # 4. 保存Tokenizer
        save_tokenizer(tokenizer, TOKENIZER_SAVE_PATH)

        # 5. 验证加载
        logger.info("\n验证Tokenizer加载...")
        from tokenizers import Tokenizer
        loaded_tokenizer = Tokenizer.from_file(TOKENIZER_SAVE_PATH)
        test_text = "验证Tokenizer加载是否正常"
        encoded = loaded_tokenizer.encode(test_text)
        decoded = loaded_tokenizer.decode(encoded.ids)
        logger.info(f"原始: {test_text}")
        logger.info(f"encoded.ids: {encoded.ids}")
        logger.info(f"编码解码后: {decoded}")
        decoded_txt=str(decoded).replace('##','').replace(" ","").replace('</w>','')
        logger.info(f"编码解码后: {decoded_txt}")
        logger.info(f"验证结果: {'✓ 成功' if test_text == decoded_txt else '✗ 失败'}")

        logger.info("\n" + "=" * 50)
        logger.info("Tokenizer训练完成！")

    except MemoryError:
        logger.error("内存不足！请尝试减小batch_size或使用更高效的提取策略")
        raise
    except Exception as e:
        logger.error(f"训练过程出现错误: {e}")
        raise


if __name__ == "__main__":
    main()




