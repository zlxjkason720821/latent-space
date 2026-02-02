# 把本地项目推送到 GitHub

## 1. 在项目根目录打开终端

进入项目文件夹（例如 `c:\Users\admin\latent space`）。

## 2. 若还没初始化 Git

```bash
git init
```

## 3. 添加远程仓库（你的 repo 地址）

把下面的地址换成你自己的（HTTPS 或 SSH 二选一）：

```bash
git remote add origin https://github.com/zlxjkason720821/latent-space.git
```

若用 SSH：
```bash
git remote add origin git@github.com:zlxjkason720821/latent-space.git
```

## 4. 添加文件、提交、推送

```bash
git add .
git status
git commit -m "Add latent space Mario project: collect, train, evaluate, play_in_latent"
git branch -M main
git push -u origin main
```

若 GitHub 要求登录，按提示用浏览器或 Personal Access Token 完成认证。

## 说明

- `.gitignore` 已配置：不会上传 `*.npz`、`saved_models/*.pt`、`results/` 下的大图和 `Super-Mario-Bros.exe`，只上传代码和说明，仓库体积小。
- 别人 clone 后需自己跑 `collect_data.py` 和 `train.py` 才能得到数据和模型；若你想把训练好的模型也放上去，可删掉 `.gitignore` 里 `saved_models/*.pt` 再 `git add` 并提交（仓库会变大）。
