# 发布规范（Releasing）

本文件规定 AeroRec 的版本标签（tag）与发布流程。**核心原则：已发布的版本号永不回收。**

---

## 1. 铁律

1. **已发布的标签不可删除、不可改指（repoint）。** 一旦某个 tag 已经推上远端并
   （可能）被 Zenodo 归档，它就是该版本的**唯一权威指针**。
2. **已发布的版本号不可复用。** 修 bug、发新版一律用**新的版本号**（`v1.0.1`、
   `v1.1.0`……），绝不"删掉 `v1.0.0` 再在别的 commit 上重建一个叫 `v1.0.0` 的 tag"。
3. **永不覆盖 `docs/paper/AeroRec-preprints.org-v1.pdf`。** v1 PDF 是论文首个版本的物证。

### 为什么第 2 条这么重要

Zenodo 依据 **GitHub Release 事件**触发归档，读取 **tag 指向 commit** 的 `.zenodo.json`。
它给每个版本分配一个**版本 DOI**（`10.5281/zenodo.XXXXXXX`）和一个**概念 DOI**
（永远指向最新版）。

如果同名 tag 被"狸猫换太子"：

- Zenodo 可能因为识别到同名 tag 的更新而**报错或产生混乱归档**；
- 已分配给旧版本的版本 DOI 与新提交内容**对不上**，引用失真；
- 读者按版本号去查，拿到的是另一份快照，**版本与内容的对应关系断裂**。

Zenodo 上归档过的 metadata **事后不可修改**，只能靠下一个版本生效——所以**错误无法撤回**。

---

## 2. 技术强制（GitHub Rulesets）

上述铁律已由两条仓库级 ruleset 强制执行，**不依赖任何人的自觉**：

| Ruleset | ID | 目标 | 规则 | 作用 |
|---|---|---|---|---|
| `published-tags-are-immutable` | `26684005` | `refs/tags/v*`、`refs/tags/preprint-*` | `deletion`、`update` | 已发布标签**不能删、不能改指** |
| `no-reuse-of-released-version-numbers` | `26684019` | `refs/tags/v1.0.0`、`refs/tags/v1.1.0`、`refs/tags/preprint-v1`、`refs/tags/preprint-v2` | `creation` | 已发布版本号**不可再次创建** |

规则集**故意不含 `creation` 通配规则**——这样未来的 `v1.2.0` 可以正常创建，
不会误伤正常发布流程（已实测验证）。

### 逃生舱（万不得已时）

规则集会拦截**所有**删除与改指，包括管理员。若确实必须动某个已发布标签
（例如一次彻底失败、从未对外引用的发布）：

1. 打开该 ruleset → Edit → 在 `exclude` 中临时加入该标签的完整 ref 名
   （如 `refs/tags/v1.0.0`）；
2. 暂时 `active` → 完成操作 → **立刻把 exclude 移除**，恢复原状；
3. 在 `CHANGELOG.md` 中如实记录这次操作与原因。

**先想清楚再动这一步。** 绝大多数情况下正确答案永远是"发一个新版本号"。

---

## 3. 发布一个新版本

```bash
# 1. 确认工作区干净、CI 全绿、审计通过
git status --short                      # 应为空
python scripts/audit_paper_numbers.py --strict
python -m pytest -q

# 2. 在 main 的最新 commit 上打标签（用仓库 owner 身份，不用临时身份）
git tag -a v1.2.0 -m "AeroRec v1.2.0：<本次变更的一句话概述>"

# 3. 推送标签（普通推送即可，不需要 --force）
git push origin v1.2.0

# 4. 创建 Release（不要 draft、不要 prerelease）
gh release create v1.2.0 --title "AeroRec v1.2.0 — <概述>" --notes-file notes.md

# 5. 等 Zenodo 自动归档，拿到两个 DOI
#    概念 DOI（永远指向最新版）/ 版本 DOI（锁定本次归档）
```

### 注意

- **不要 `--force` 推送标签。** 若 `git push` 报 `would clobber existing tag`，
  说明本地标签与远端不一致——**先查清原因**（多半是本地残留了一个重建过的
  旧副本），用 `git fetch origin refs/tags/<name>` 取回远端权威对象对齐，
  而不是强行覆盖。
- **Zenodo 读取的是 tag 指向 commit 的 `.zenodo.json`**，不是 main 分支的。
  所以 `.zenodo.json` 必须在**打标签之前**就定稿——归档后改它对本次记录无效。
- Zenodo 会用 tag 名覆盖 `.zenodo.json` 里的 `version`，并自行生成
  `doi` / `publication_date` / `created` / `modified`，**不要手写这些字段**。
- 本地标签身份请用仓库 owner：`Elijah-Lin <linyvlin718@gmail.com>`。
  用临时身份（如 `agent@local`）打标签会造出与远端不同的 tag 对象，
  徒增"同名不同物"的混乱风险。

---

## 4. 标签含义

| 标签 | 指向 | 含义 |
|---|---|---|
| `v0.1.0` / `v0.2.0` | 历史 commit | 开发期里程碑，从未发布 Release |
| `preprint-v1` | `c41997b` | 论文 **v1** 产出时的仓库状态（数字 `0.3576` 的历史快照） |
| `preprint-v2` | `d187766` | 论文 **v2**（更正 `0.3576 → 0.3625`）定版时的仓库状态 |
| `v1.0.0` | `37c0727` | 首个计划发布版本；**从未真正发布过 Release**，Zenodo 上无对应记录 |
| `v1.1.0` | `8aee4f9` | 首个真实 Release；Zenodo 归档为 `10.5281/zenodo.24003811` |

`preprint-v1` / `preprint-v2` 是**论文状态的冻结锚点**，与软件版本号是两套独立
编号体系，同样受不可变规则保护。

---

## 5. 版本号与 DOI 对应

| 论文版本 | DOI | 软件版本 | Zenodo 版本 DOI |
|---|---|---|---|
| v1 | `10.20944/preprints202610.2582.v1` | — | — |
| v2 | `10.20944/preprints202610.2582.v2` | `v1.1.0` | `10.5281/zenodo.24003811` |

- 论文 v1 的 DOI **永久保留**并继续可访问，不因 v2 发布而失效。
- 论文 v2 与软件 `v1.1.0` 是**同一次更正的两种产物**（论文 + 代码），
  故 `preprint-v2` 与 `v1.1.0` 的归档内容一致（`data/` 与 `preprint-v1`
  逐字节相同，未重跑任何实验）。
- 概念 DOI `10.5281/zenodo.24003810` 永远指向最新版引用它的人。
