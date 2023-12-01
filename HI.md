```bash
git branch    # 查看所有本地分支及当前所在分支（带 * 号的是当前分支）
git branch <branch-name>    # 创建名为 <branch-name> 的新分支
git checkout <branch-name>    # 切换到名为 <branch-name> 的分支
git checkout -b <branch-name>    # 创建并切换到名为 <branch-name> 的新分支

git add . # 提交到暂存区
git commit -m '*****' # 上传2
git push origin <branch-name> #远程
```

流程:
```bash
git branch    # 查看所有本地分支及当前所在分支（带 * 号的是当前分支）
git checkout master    # 切换到master分支
git pull # 拉取最新信息
git checkout -b <branch-name>    # 创建并切换到名为 <branch-name> 的新分支


git add . # 提交到暂存区
git commit -m '*****' # 上传2
git push origin <branch-name> #远程


```