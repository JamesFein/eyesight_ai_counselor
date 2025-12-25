# 提示词文档说明

本目录包含眼科近视防控 AI 咨询系统的所有提示词文件。系统采用两阶段搜索架构，通过多个专业 Agent 协同工作，为用户提供精准、可溯源的问答服务。

---

## 目录结构

```
prompts/
├── README.md                              # 本文档
├── decomposition_system_prompt.txt        # 问题分解Agent - 系统提示词
├── decomposition_user_prompt.txt          # 问题分解Agent - 用户提示词
├── relevance_judge_system_prompt.txt      # 相关性判断Agent - 系统提示词
├── relevance_judge_user_prompt.txt        # 相关性判断Agent - 用户提示词
├── sufficiency_judge_system_prompt.txt    # 充分性判断Agent - 系统提示词
├── sufficiency_judge_user_prompt.txt      # 充分性判断Agent - 用户提示词
├── answer_generation_system_prompt.txt    # 答案生成Agent - 系统提示词
├── answer_generation_user_prompt.txt      # 答案生成Agent - 用户提示词
├── query_classifier_system_prompt.txt     # 问题分类器 - 系统提示词
├── query_classifier_user_prompt.txt       # 问题分类器 - 用户提示词
├── query_rewrite_system_prompt.txt        # 问题改写Agent - 系统提示词
├── query_rewrite_user_prompt.txt          # 问题改写Agent - 用户提示词
├── intent_classifier_system_prompt.txt    # 意图识别Agent - 系统提示词
├── intent_classifier_user_prompt.txt      # 意图识别Agent - 用户提示词
├── simple_answer_system_prompt.txt        # 简单答案生成 - 系统提示词
└── simple_answer_user_prompt.txt          # 简单答案生成 - 用户提示词
```

---

## 核心 Agent（PRD 要求）

### 1. 问题分解 Agent (Decomposition Agent)

**功能**：将用户问题分解为 1-5 个子检索肯定句（非问句），用于在文档库中精准检索

**模型建议**：gpt-4.1 / gpt-4.1（需要批判性思维）

**文件**：

- 系统提示词：`decomposition_system_prompt.txt`
- 用户提示词：`decomposition_user_prompt.txt`

**输入变量**：

```python
{
    "original_query": "用户的原始问题"
}
```

**输出结构**：

```json
{
  "sub_queries": [
    {
      "id": "sq_1",
      "statement": "子检索肯定句内容",
      "focus": "聚焦点（1-2个词）"
    }
  ],
  "reasoning": "分解理由"
}
```

**关键特点**：

- 使用肯定句而非问句（更贴近文档原文）
- 覆盖多维度：定义、原理、适应症、禁忌症、效果、注意事项等
- 使用专业术语和同义词
- 简单问题 1-2 个，复杂问题最多 5 个

---

### 2. 相关性判断 Agent (Relevance Judge Agent)

**功能**：判断检索到的文本块相关性，提取可用的原文段落，生成子答案

**模型建议**：gpt-4.1 / gpt-4.1（需要批判性思维）

**文件**：

- 系统提示词：`relevance_judge_system_prompt.txt`
- 用户提示词：`relevance_judge_user_prompt.txt`

**输入变量**：

```python
{
    "sub_query_id": "sq_1",
    "sub_query_statement": "子检索肯定句",
    "sub_query_focus": "聚焦点",
    "retrieved_chunks": [
        {
            "source_file": "文件名.md",
            "score": 0.85,
            "content": "文本块内容"
        }
    ]
}
```

**输出结构**：

```json
{
  "sub_query_id": "sq_1",
  "sub_query_statement": "原始子检索肯定句",
  "relevant_extracts": [
    {
      "source_file": "来源文件名.md",
      "excerpt": "提取的原文段落（保持原文不改写）",
      "relevance": "high/medium/low"
    }
  ],
  "sub_answer": "基于提取段落的简洁子答案（1-3句话）",
  "confidence": "high/medium/low",
  "missing_aspects": ["缺失的知识点1", "缺失的知识点2"]
}
```

**关键特点**：

- 保持原文表述，不改写
- 标注来源文件名
- 评估置信度
- 识别缺失知识点

---

### 3. 充分性判断 Agent (Sufficiency Judge Agent)

**功能**：判断第一阶段检索是否充分，决定是否需要第二阶段

**模型建议**：gpt-4.1 / gpt-4.1（需要批判性思维）

**文件**：

- 系统提示词：`sufficiency_judge_system_prompt.txt`
- 用户提示词：`sufficiency_judge_user_prompt.txt`

**输入变量**：

```python
{
    "original_query": "用户原始问题",
    "sub_queries": [
        {"id": "sq_1", "statement": "...", "focus": "..."}
    ],
    "sub_query_analyses": [
        {
            "sub_query_id": "sq_1",
            "sub_answer": "...",
            "confidence": "high",
            "relevant_extracts": [...],
            "missing_aspects": [...]
        }
    ]
}
```

**输出结构**：

```json
{
  "is_sufficient": true/false,
  "confidence": "high/medium/low",
  "covered_aspects": ["已覆盖的方面1", "已覆盖的方面2"],
  "missing_critical_aspects": ["缺失的关键方面1"],
  "reasoning": "判断理由（2-3句话）",
  "additional_sub_queries": [
    {
      "id": "sq_add_1",
      "statement": "追加的子检索肯定句",
      "focus": "聚焦点",
      "reason": "为什么需要这个追加检索"
    }
  ]
}
```

**关键特点**：

- 扮演严谨验光师角色
- 目标触发率 < 20%（大多数情况一阶段足够）
- 追加子检索最多 2-3 条
- 只在确实缺少关键信息时才触发第二阶段

---

### 4. 答案生成 Agent (Answer Generation Agent)

**功能**：流式生成结构化小型报告（TL;DR → 完整结论 → 论据与来源）

**模型建议**：gpt-4.1（高质量输出）

**文件**：

- 系统提示词：`answer_generation_system_prompt.txt`
- 用户提示词：`answer_generation_user_prompt.txt`

**输入变量**：

```python
{
    "original_query": "用户原始问题",
    "sub_query_analyses": [
        {
            "sub_query_statement": "...",
            "sub_answer": "...",
            "confidence": "high",
            "relevant_extracts": [
                {
                    "source_file": "文件名.md",
                    "excerpt": "原文段落"
                }
            ]
        }
    ]
}
```

**输出格式**：

```markdown
## TL;DR

[1-2 句话高度概括核心答案]

## 完整结论

[2-4 句话的专业完整结论]

## 论据与来源

### 1. [论点标题]

[论点说明]

> 来源：《文档名》
> "[原文摘录]"

### 2. [论点标题]

[论点说明]

> 来源：《文档名》
> "[原文摘录]"
```

**关键特点**：

- 结论优先输出
- 每个论点必须有来源支持
- 引用原文保持原文表述
- 使用通俗易懂的语言
- 流式输出

---

## 辅助 Agent（系统增强）

### 5. 意图识别 Agent (Intent Classifier)

**功能**：守门员，判断用户问题是否在服务范围内

**模型建议**：gpt-4.1（快速分类）

**文件**：

- 系统提示词：`intent_classifier_system_prompt.txt`
- 用户提示词：`intent_classifier_user_prompt.txt`

**输入变量**：

```python
{
    "user_query": "用户问题",
    "conversation_history": [  # 可选
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

**输出结构**：

```json
{
  "intent": "in_scope/edge_case/out_of_scope/greeting/unclear",
  "confidence": "high/medium/low",
  "category": "问题所属类别（如：OK镜、阿托品等）",
  "reasoning": "判断理由（1句话）",
  "suggested_response": "如果out_of_scope或greeting，提供建议回复"
}
```

**服务范围**：

- in_scope：近视防控方法、近视知识、眼科检查、儿童青少年视力健康
- edge_case：成人近视、其他眼部问题的基础知识
- out_of_scope：非眼科问题、与近视防控无关的眼科问题
- greeting：问候语、闲聊
- unclear：问题不清晰

---

### 6. 问题改写 Agent (Query Rewrite Agent)

**功能**：多轮对话中的指代消解和省略补充

**模型建议**：gpt-4.1（快速改写）

**文件**：

- 系统提示词：`query_rewrite_system_prompt.txt`
- 用户提示词：`query_rewrite_user_prompt.txt`

**输入变量**：

```python
{
    "current_query": "用户当前问题",
    "conversation_history": [  # 可选
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]
}
```

**输出结构**：

```json
{
  "original_query": "用户原始问题",
  "rewritten_query": "改写后的完整问题",
  "needs_rewrite": true/false,
  "rewrite_reason": "改写原因（如果需要改写）"
}
```

**改写规则**：

- 指代消解（"它"→ 具体对象）
- 省略补充（补充主语、条件等）
- 保持原意
- 简洁完整

---

### 7. 问题分类器 (Query Classifier)

**功能**：判断简单/复杂问题，决定使用快速路径还是标准路径

**模型建议**：gpt-4.1（快速分类）

**文件**：

- 系统提示词：`query_classifier_system_prompt.txt`
- 用户提示词：`query_classifier_user_prompt.txt`

**输入变量**：

```python
{
    "original_query": "用户问题"
}
```

**输出结构**：

```json
{
  "query_type": "simple/complex",
  "confidence": "high/medium/low",
  "reasoning": "判断理由（1句话）",
  "suggested_search_term": "如果是简单问题，建议的检索关键词"
}
```

**简单问题特征**：

- 单一概念查询（"XX 是什么？"）
- 直接事实查询（"XX 的正常范围是多少？"）
- 是非判断查询（"XX 能不能 XX？"）
- 单一维度查询

**复杂问题特征**：

- 比较选择类（"A 和 B 哪个更好？"）
- 个性化建议类（包含年龄、度数等信息）
- 多维度综合类
- 决策支持类
- 流程/方法类

---

### 8. 简单答案生成 Agent (Simple Answer Generation)

**功能**：快速路径的简洁答案生成（不走两阶段流程）

**模型建议**：gpt-4.1（快速生成）

**文件**：

- 系统提示词：`simple_answer_system_prompt.txt`
- 用户提示词：`simple_answer_user_prompt.txt`

**输入变量**：

```python
{
    "original_query": "用户问题",
    "retrieved_chunks": [
        {
            "source_file": "文件名.md",
            "score": 0.85,
            "content": "文本块内容"
        }
    ]
}
```

**输出格式**：

```
[直接回答内容，2-4句话]

——来源：《文档名》
```

**关键特点**：

- 简洁直接（2-4 句话）
- 专业且通俗
- 标注来源
- 必要时安全提醒

---

## 完整处理流程

```
┌─────────────┐
│  用户输入   │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│  意图识别Agent  │
└────────┬────────┘
         │
         ├──► out_of_scope/greeting ──► 直接回复
         │
         ▼ in_scope
┌─────────────────┐
│  问题改写Agent  │ (多轮对话时)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  问题分类器     │
└────────┬────────┘
         │
         ├──► simple ──► 单次检索 + 简单答案生成 (快速路径)
         │
         ▼ complex
┌─────────────────┐
│ 问题分解Agent   │ → 生成1-5个子检索肯定句
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  并行混合检索   │ (BM25 + 语义检索)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 相关性判断Agent │ (并行处理所有子检索)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 充分性判断Agent │
└────────┬────────┘
         │
         ├──► insufficient ──► 第二阶段检索 (追加2-3个子检索)
         │                     │
         │                     ▼
         │              重复混合检索 + 相关性判断
         │                     │
         ▼◄────────────────────┘
┌─────────────────┐
│ 答案生成Agent   │ (流式输出)
└────────┬────────┘
         │
         ▼
   结构化报告输出
```

---

## 模板语法说明

user_prompt 文件使用**Handlebars 模板语法**：

### 基本变量

```handlebars
{{variable_name}}
```

### 条件判断

```handlebars
{{#if condition}}
  内容
{{else}}
  其他内容
{{/if}}
```

### 数组遍历

```handlebars
{{#each array_name}}
  {{this.property}}
  {{@index}}
  <!-- 数组索引 -->
{{/each}}
```

---

## 配置建议

### Agent 模型配置（.env）

```ini
# 问题分解Agent
DECOMPOSITION_MODEL=gpt-4.1
DECOMPOSITION_TEMPERATURE=0.3
DECOMPOSITION_MAX_TOKENS=800

# 相关性判断Agent
RELEVANCE_MODEL=gpt-4.1
RELEVANCE_TEMPERATURE=0.2
RELEVANCE_MAX_TOKENS=1000

# 充分性判断Agent
SUFFICIENCY_MODEL=gpt-4.1
SUFFICIENCY_TEMPERATURE=0.3
SUFFICIENCY_MAX_TOKENS=500

# 答案生成Agent
ANSWER_MODEL=gpt-4.1
ANSWER_TEMPERATURE=0.5
ANSWER_MAX_TOKENS=2000

# 辅助Agent
INTENT_MODEL=gpt-4.1
QUERY_REWRITE_MODEL=gpt-4.1
CLASSIFIER_MODEL=gpt-4.1
SIMPLE_ANSWER_MODEL=gpt-4.1
```

### 检索参数配置

```ini
# 子检索数量限制
MAX_SUB_QUERIES=5
MAX_ADDITIONAL_QUERIES=3

# 检索参数
BM25_TOP_K=5
SEMANTIC_TOP_K=5
BM25_THRESHOLD=0.0
SEMANTIC_THRESHOLD=0.5
```

---

## 性能优化要点

### 时间分配目标

| 阶段             | 目标耗时   | 优化手段          |
| ---------------- | ---------- | ----------------- |
| 意图识别         | ~0.3 秒    | gpt-4.1           |
| 问题改写         | ~0.3 秒    | gpt-4.1           |
| 问题分类         | ~0.3 秒    | gpt-4.1           |
| 问题分解         | ~1 秒      | gpt-4.1 + 缓存    |
| 第一阶段并行检索 | ~1.5 秒    | asyncio.gather    |
| 第一阶段并行判断 | ~1.5 秒    | asyncio.gather    |
| 充分性判断       | ~0.5 秒    | gpt-4.1           |
| 答案生成启动     | 立即       | 流式输出          |
| **首字延迟**     | **< 5 秒** | **总计约 4.5 秒** |

### 并行化策略

1. **所有子检索完全并行**：使用 `asyncio.gather` 并行执行所有子检索
2. **BM25 和语义检索并行**：每个子检索内部也并行
3. **所有相关性判断并行**：同时处理所有子检索的相关性判断
4. **第二阶段同样并行**：如果触发，采用相同并行策略

---

## 提示词维护指南

### 修改原则

1. **保持角色定位一致**：每个 Agent 的专业定位不变
2. **输出格式稳定**：JSON 结构变化需同步更新代码
3. **示例保持更新**：修改规则后更新示例
4. **版本控制**：重大修改时备份旧版本

### 优化方向

1. **减少 token 消耗**：简化示例，移除冗余说明
2. **提高准确性**：根据实际表现调整判断标准
3. **增强鲁棒性**：添加边界情况处理说明
4. **改进可读性**：使用清晰的结构和标记

### 测试建议

1. **单元测试**：每个 Agent 独立测试
2. **集成测试**：完整流程测试
3. **边界测试**：异常输入、空值、极端情况
4. **A/B 测试**：对比不同提示词版本的效果

---

## 常见问题

### Q1: 如何调整第二阶段触发率？

修改 `sufficiency_judge_system_prompt.txt` 中的：

- 判断标准的严格程度
- 触发率目标描述（当前是 < 20%）
- 示例中充分/不充分的比例

### Q2: 如何让答案更简洁/更详细？

修改 `answer_generation_system_prompt.txt` 中的：

- 输出长度要求（当前是 2-4 句话）
- 论据数量（当前是 3-5 个）
- 详细程度描述

### Q3: 如何添加新的问题类型？

1. 在 `query_classifier_system_prompt.txt` 添加类型定义
2. 在 `decomposition_system_prompt.txt` 添加该类型的分解策略
3. 添加对应的示例

### Q4: 如何扩展服务范围？

修改 `intent_classifier_system_prompt.txt` 中的：

- 核心范围列表
- 边缘范围列表
- 不在范围内列表

---

## 版本记录

| 版本 | 日期       | 说明                                |
| ---- | ---------- | ----------------------------------- |
| v1.0 | 2025-12-25 | 初始版本，包含 8 组 16 个提示词文件 |

---

## 联系方式

如有问题或建议，请联系项目维护者。
