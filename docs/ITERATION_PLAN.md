# 迭代计划：0.3576 更正（论文 v2 + Zenodo 归档）

> 状态：进行中 · 建立于 2026-11-05
> 目的：修掉预印本里唯一一处无法从仓库复现的数字（`0.3576`），并让论文、仓库、归档三者的数字与 DOI 完全对齐。

## 0. 起因

预印本 §4.3 与 Fig. 3 图注写"边界在小规模配置下为 0.3576，扩规模后为 0.3579，因此对规模稳健"。
`0.3576` 来自早期实现的某次运行，当时未落盘；当前代码按同配置
（`boundary_by_range(t_pop=0.25, n_records=600, n_seeds=30)`）复算得 **0.3625**。
`data/bias/boundary_sweep.json` 里出现的 `0.3576` 是 τ=2.0 那档的 naive RMSE，与边界值无关，纯属撞数字。

定性：**不是编造，是可追溯性缺口**。错在"把一个没有产物支撑的数字当最终值写进论文"，不在代码。
结论（边界对规模稳健）不受影响，但措辞里的 "only" 与 1.3% 的实际变化不符，需一并改。

## 1. 三条决策要点

- **不覆盖 v1 PDF**：`docs/paper/AeroRec-preprints.org-v1.pdf` 是 `0.3576` 的物证，原地保留；v2 新增为独立文件。
- **不先 push 再等审核**：v2 有可能被编辑打回，且正式 PDF 只有平台上线后才存在。仓库只在审核通过后动一次。
- **`preprint-v1` 标签不动**：它钉在 `c41997b`，诚实记录 v1 的产出状态。改完只能新增 `preprint-v2`。

## 2. 阶段与待办

### 阶段 1 — 本地改论文与登记文件（不 commit）

- [x] 论文 §4.3 正文：`0.3576` → `0.3625`，删去 `only`，改为如实描述 1.3% 的变化
      *（实际处理：复算发现小规模与大规模落在同一网格档 π_ratio=40，故保留 `only` 但补上"0.005 vs 0.29 间距"的量化依据，比删掉更有据）*
- [x] 论文 Fig. 3 图注：`(0.3576 → 0.3579)` → 带变化幅度的表述
- [x] `scripts/audit_paper_numbers.py`：基线改 `0.3625`，清空 `EXPECTED_GAPS`，改写注释与 docstring
      *（同时修掉一处设计缺陷：原逻辑"差 < 0.01 自动登记为已知差异"等于脚本给自己发豁免，已改为严格比对 + 2e-4 容差）*
- [x] `docs/paper/PROVENANCE.md` §5：由"已知差异"改写为"已更正"，记录更正前后值
- [x] `docs/paper/PROVENANCE.md` §7：修正 `v1.0.0` 的指向（`1fc5e0f` → `37c0727`）
- [x] 跑 `audit_paper_numbers.py`，确认 207/207 全一致、0 豁免（`--strict` 退出码 0；ruff / mypy 均通过）

### 阶段 2 — 生成 v2 上传稿（不 commit）

- [x] 确认 preprints.org 接受的源文件格式（Word / LaTeX / PDF）
      *（公开的 Instructions for Authors 未列出格式白名单，只说经编辑筛查；Word 是最通用的选项，故产出 .docx，平台侧若要求 LaTeX 可再转）*
- [x] 按平台要求生成 v2 上传稿，图与表随正文 → `docs/paper/AeroRec-preprints.org-v2-draft.docx`（978 KB，13 页当量）
      *（新增 `scripts/build_preprint_docx.py`：<sup> 引用转上标、图限宽 15.5 cm、8 张表加边框并降为 8 pt、图注居中）*
- [x] 自检：正文、图注、表格中的边界值三处一致；无残留 `0.3576`
      *（docx 内 0.3625 ×2、0.3576 ×0；11 处引用均为上标；8 表总宽 13.97 cm < 版心 16 cm；6 图齐全）*

### 阶段 3 — 提交预印本 v2（用户执行）

- [x] 在 preprints.org 提交新版本，附更正说明（英文，一段话说清改了什么、为什么、结论是否受影响）
      *（2026-11-05 02:19:36 提交，Preprints ID `242644`，状态 `Pending Check`；已确认上传稿内 `0.3576` 归零、`0.3625` 就位）*
- [ ] 等待审核（约 24 小时）

### 阶段 4 — 取回 v2 产出（用户执行）

- [x] 上线后提供新的 `.v2` DOI → `10.20944/preprints202610.2582.v2`
- [x] 下载平台回传的正式 PDF → `docs/paper/AeroRec-preprints.org-v2.pdf`（14 页，1.7 MB）

> **结果（2026-11-06）**：v2 通过审核并上线，平台排版 PDF 已入库。入库前核对：
> 6 图与 `docs/paper/figures/` 逐张对应（MAE 0.07–0.14，次优 ≥ 10）、8 表齐全、
> `0.3625` ×2 / `0.3576` ×0。DOI 尚在平台侧生效中（当晚至次日），链接按最终形态写死。

### 阶段 5 — 仓库落地

> **已提前执行（2026-11-05）**：原计划等 v2 上线后一次性提交，实际按用户要求先把代码与文档落地，
> 避免本地改动静默堆着。DOI 与正式 PDF 留到 v2 上线后补一次提交（5b）。

**5a — 代码与文档落地（已提交）**

- [x] `scripts/audit_paper_numbers.py` 更正基线、清空豁免
- [x] `docs/paper/PROVENANCE.md` 改写第 5 节、修正第 7 节
- [x] 新增 `scripts/build_preprint_docx.py`（v2 排版脚本）
- [x] 新增 `docs/paper/AeroRec-preprints.org-v2-draft.docx`（上传稿）
- [x] 新增 `docs/ITERATION_PLAN.md`（本文件）
- [x] commit + push，`preprint-v1` 与 `v1.0.0` 保持原位

**5a-补 — 源文入库与 CI 门禁（已提交）**

- [x] 论文源文入库为 `docs/paper/manuscript.md`（带 v2 版本说明注释），成为唯一权威副本
- [x] `audit_paper_numbers.py` 改读仓库内副本；仓库外若留有不一致的工作副本则直接失败并打印 diff
- [x] `.github/workflows/repro.yml` 的 `bias-experiments` job 挂上 `audit_paper_numbers.py --strict`
      *（此前 `--strict` 只是能力，没有任何地方调用；现在每次 push 都会真的校验论文数字）*
- [x] `docs/paper/README.md` 清单收录 `manuscript.md`；`PROVENANCE.md` 第 4 节说明 CI 门禁

**5b — v2 上线后收尾（已完成）**

- [x] 放入正式 PDF 为 `docs/paper/AeroRec-preprints.org-v2.pdf`，删除 `-draft` 后缀的中间稿
      *（中间稿可由 `scripts/build_preprint_docx.py` 从 `manuscript.md` 确定性重生成，故不入库）*
- [x] DOI 全量替换：`CITATION.cff`、`.zenodo.json`、`README.md`、`README.zh.md`、`docs/index.html`、
      三份 docs、`app.py`、`src/serve/i18n.py`、`PROVENANCE.md`、`docs/paper/README.md`（共 13 个文件）
- [x] `docs/paper/README.md` 材料清单：v2 标为 version of record，v1 标为 superseded（保留为 0.3576 的物证）
- [x] `CHANGELOG.md` 记录 v2 上线与 DOI 替换；`CITATION.cff` / `.zenodo.json` 升版 1.1.0
- [ ] 打标签 `preprint-v2` 与 `v1.1.0`
      *（`preprint-v1` → `c41997b`、`v1.0.0` → `37c0727` 保持不动）*

### 阶段 6 — Zenodo 归档（用户执行）

- [ ] 用标签 **`v1.1.0`** 发 GitHub Release（标题/描述已备好，见下）
- [ ] Zenodo 自动归档，取得软件 DOI

> Release 标题：`AeroRec v1.1.0 — preprint v2 correction (boundary 0.3576 → 0.3625)`
> 描述要点：结果文件与 v1 归档逐字节相同；变的只有论文一个值、审计基建与文档。
> 归档后 Zenodo 记录会生成**新**的软件 DOI（与 v1.0.0 那条独立），两者靠
> `related_identifiers` 互链。

### 阶段 7 — 补软件 DOI（1 个 commit）

- [ ] 两版 README 徽章并列预印本 DOI 与软件 DOI
- [ ] `docs/index.html` 页脚
- [ ] `CHANGELOG.md`
- [ ] `.zenodo.json` 的 `related_identifiers` 指向 `.v2`（已指向，待补软件 DOI 互链）

## 3. 红线

1. `docs/paper/AeroRec-preprints.org-v1.pdf` 不得覆盖、不得删除。
2. `preprint-v1` 标签不得移动。
3. `data/bias/*.json` 与 `data/real_validation/*.json` 不得改动（论文数值底座）。
4. 不得为了"对上"而改代码去凑 `0.3576`。
5. 阶段 5 的 commit 必须是 Zenodo 归档前最后一次改动。

## 4. 验收

- `python scripts/audit_paper_numbers.py --strict` 退出码 0，且无豁免项。
- 论文正文、图注、表 3 三处的边界值互相自洽。
- 仓库内 `0.3576` 仅出现在历史记录（`CHANGELOG`、本文件、v1 PDF）中，不再出现在任何"当前值"位置。
- GitHub Release、Zenodo 记录、`CITATION.cff`、两份 README 的 DOI 指向同一个版本。
