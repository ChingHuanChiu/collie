# PyPI Release Checklist

使用此清單確保在發布到 PyPI 之前一切準備就緒。

## Pre-Release 檢查

### 1. 代碼質量
- [x] 所有核心模組無語法錯誤
- [x] 移除所有 `print()` 語句，使用 logger
- [x] 類型提示完整
- [x] 文檔字符串完整

### 2. 版本管理
- [x] `collie/__init__.py` 中的 `__version__` 已更新
- [x] `pyproject.toml` 中的 version 已更新
- [x] `setup.cfg` 中的 version 已更新
- [x] 創建 `CHANGELOG.md` 並記錄此版本的變更

### 3. 文檔
- [x] README.md 完整且最新
- [x] 安裝說明清晰
- [x] 使用範例可運行
- [x] API 文檔已生成
- [x] LICENSE 文件存在

### 4. 配置文件
- [x] `pyproject.toml` 配置正確
  - [x] 依賴項列表完整
  - [x] classifiers 正確
  - [x] Python 版本要求正確 (>=3.10)
- [x] `MANIFEST.in` 包含所有必要文件
- [x] `.gitignore` 排除不必要的文件

### 5. 測試
- [ ] 在乾淨的虛擬環境中測試安裝
- [ ] 所有範例代碼可運行
- [ ] 文檔中的代碼片段正確

### 6. 依賴項
- [x] 所有必需的依賴項都在 `dependencies` 中
- [x] 依賴項版本範圍合理
- [x] 移除了開發依賴（如 pytest, black 等）

### 7. Package 結構
- [x] `collie/__init__.py` 正確導出所有公共 API
- [x] `__all__` 列表完整
- [x] 無循環導入

## Build 測試

### 8. 本地構建測試

```bash
# 1. 清理舊的構建文件
rm -rf dist/ build/ *.egg-info

# 2. 構建 package
python -m build

# 3. 檢查構建的檔案
ls -lh dist/

# 4. 檢查 package 內容
tar -tzf dist/collie-mlops-0.1.0b0.tar.gz | head -50

# 5. 使用 twine 檢查
python -m twine check dist/*
```

### 9. 本地安裝測試

```bash
# 在新的虛擬環境中測試
python -m venv test_env
source test_env/bin/activate  # macOS/Linux
# 或
# test_env\Scripts\activate  # Windows

pip install dist/collie_mlops-0.1.0b0-py3-none-any.whl

# 測試導入
python -c "from collie import Transformer, Trainer, Orchestrator; print('Import successful!')"

# 運行簡單測試
python -c "
from collie import Event, TransformerPayload
event = Event(payload=TransformerPayload(train_data=None))
print(f'Event created: {event}')
"

deactivate
```

## TestPyPI 發布

### 10. 上傳到 TestPyPI

```bash
# 確保有 TestPyPI 帳號和 token

# 上傳到 TestPyPI
python -m twine upload --repository testpypi dist/*

# 從 TestPyPI 安裝測試
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ collie-mlops
```

### 11. TestPyPI 驗證
- [ ] Package 頁面顯示正確
- [ ] README 渲染正常
- [ ] 元數據（作者、授權等）正確
- [ ] 依賴項正確安裝
- [ ] 可以成功導入和使用

## 正式發布到 PyPI

### 12. 最終檢查
- [ ] 在 TestPyPI 上完成所有測試
- [ ] Git repository 已清理
- [ ] 創建 git tag：`git tag v0.1.0-beta`
- [ ] Push tag：`git push origin v0.1.0-beta`

### 13. 發布到 PyPI

```bash
# 上傳到正式 PyPI
python -m twine upload dist/*
```

### 14. 發布後驗證
- [ ] 訪問 https://pypi.org/project/collie-mlops/
- [ ] 檢查 package 頁面
- [ ] 測試安裝：`pip install collie-mlops`
- [ ] 運行範例代碼

## 發布後任務

### 15. GitHub Release
- [ ] 在 GitHub 創建 Release
- [ ] 附上 CHANGELOG
- [ ] 標記版本號

### 16. 文檔更新
- [ ] 更新 ReadTheDocs（如果使用）
- [ ] 更新 GitHub README badges
- [ ] 公告發布（Twitter, LinkedIn 等）

### 17. 社群
- [ ] 準備公告文章
- [ ] 回應初期用戶反饋
- [ ] 監控 GitHub issues

## 常見問題

### Q: 版本號格式
- Beta 版本: `0.1.0-beta` 或 `0.1.0b0`
- 正式版本: `0.1.0`
- 修補版本: `0.1.1`

### Q: 如果上傳錯誤怎麼辦？
- **無法刪除已上傳的版本**
- 必須發布新版本（例如 0.1.0-beta2）
- 因此務必先在 TestPyPI 測試

### Q: 需要什麼帳號？
- PyPI 帳號: https://pypi.org/account/register/
- TestPyPI 帳號: https://test.pypi.org/account/register/
- 需要為每個帳號創建 API token

## 有用的命令

```bash
# 安裝構建工具
pip install --upgrade build twine

# 檢查 setup.py
python setup.py check

# 查看 package 元數據
python -m build --sdist
tar -xzf dist/collie-mlops-*.tar.gz
cat collie-mlops-*/PKG-INFO

# 清理構建文件
rm -rf build/ dist/ *.egg-info __pycache__
find . -type d -name __pycache__ -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
```

## 最後提醒

- ✅ TestPyPI 測試是必須的
- ✅ 版本號一旦發布無法修改
- ✅ 確保所有敏感信息已移除
- ✅ 檢查 LICENSE 文件正確
- ✅ README 在 PyPI 上正確渲染
