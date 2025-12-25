## 特别提醒：Windows 文件路径

**重要：本项目在 Windows 环境下运行，存在已知的文件操作 bug。**

**强制要求：**

- 使用 Read、Write、Edit、NotebookEdit 等文件操作工具时，**必须使用 Windows 绝对路径格式**
- 路径格式：`C:\Users\Administrator\Desktop\eye_app\文件名.md`
- **禁止使用相对路径**（如 `示例用户原始数据.md`）
- **禁止使用正斜杠路径**（如 `C:/Users/Administrator/Desktop/eye_app/文件名.md`）

**示例：**

```python
# ✅ 正确 - Windows 绝对路径
Read(file_path="C:\Users\Administrator\Desktop\eye_app\ai眼科报告需求.md")
Edit(file_path="C:\Users\Administrator\Desktop\eye_app\CLAUDE.md", ...)

# ❌ 错误 - 相对路径
Read(file_path="ai眼科报告需求.md")

# ❌ 错误 - 正斜杠路径
Read(file_path="C:/Users/Administrator/Desktop/eye_app/ai眼科报告需求.md")
```

## 特别提醒：Windows 环境命令兼容性

**重要：本项目运行在 Windows 平台（Platform: win32），必须使用跨平台兼容的命令。**

### Bash 命令使用规范

**禁止使用的 Linux 特有命令：**

- ❌ `tree` - Windows 默认不支持
- ❌ `grep`、`rg` - 使用 Grep 工具代替
- ❌ `find`（复杂用法）- 优先使用 Glob 工具

**Windows 环境推荐命令：**

- ✅ `dir` - 列出文件和目录
- ✅ `dir /s /b /ad` - 递归列出所有目录
- ✅ `ls` - Git Bash 环境支持
- ✅ `find . -type d -maxdepth 2` - Git Bash 环境支持的简单 find 用法

**最佳实践：**

- 查找文件：优先使用 **Glob 工具**而非 `find` 命令
- 搜索内容：优先使用 **Grep 工具**而非 `grep`/`rg` 命令
- 查看目录：使用 `ls` 或 `dir` 而非 `tree`

### Edit 工具使用规范

**强制流程：使用 Edit 工具前必须先 Read 文件！**

1. **第一步：使用 Read 工具读取目标区域**

   ```python
   Read(file_path="C:\Users\Administrator\Desktop\eye_app\CLAUDE.md", offset=337, limit=10)
   ```

2. **第二步：精确复制要替换的文本作为 old_string**

   - 必须包含**所有空格、标点符号、换行符**
   - 不能有任何多余或缺少的字符
   - 建议使用 Read 工具输出的原文直接复制

3. **第三步：确认匹配后再调用 Edit**

   ```python
   Edit(
       file_path="C:\Users\Administrator\Desktop\eye_app\CLAUDE.md",
       old_string="需对比历史数据，需对比历史数据",  # ✅ 精确匹配（包含逗号）
       new_string="需对比历史数据进行综合判断"
   )
   ```

**常见错误示例：**

```python
# ❌ 错误 - 未先 Read 就直接 Edit
Edit(file_path="...", old_string="猜测的内容", new_string="...")

# ❌ 错误 - 标点符号不匹配
old_string="需对比历史数据"  # 缺少逗号
实际文本="需对比历史数据，"  # 有逗号

# ❌ 错误 - 空格数量不对
old_string="1. **年龄依赖型数据**"  # 1个空格
实际文本="1.  **年龄依赖型数据**"  # 2个空格
```

**Edit 工具失败时的应对：**

- 立即使用 Read 工具重新读取该区域
- 对比 Read 输出与你的 old_string，找出差异
- 使用完全一致的文本重试