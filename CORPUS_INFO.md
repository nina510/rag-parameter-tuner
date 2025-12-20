# Corpus 目录说明

## 目录结构

```
corpus/
├── csv/
│   └── LongCovid.csv          # 文档元数据CSV文件
└── txt/
    └── LongCovid/
        ├── AAPMRCompendium_NP002.txt
        ├── Al-Aly_39122965.txt
        ├── Bateman_34454716.txt
        ├── Fineberg_39110819.txt
        ├── Mueller_40105889.txt
        ├── Peluso_39326415.txt
        ├── Vogel_39142505.txt
        └── Zeraatkar_39603702.txt
```

## 用途

这个 `corpus` 目录包含了8篇核心 Long COVID 文献的文本文件，用于支持自定义 chunk 参数功能。

当用户选择自定义 `chunk_size` 和 `chunk_overlap` 参数时，系统会：
1. 从 `corpus/txt/LongCovid/` 加载这8篇核心文献的文本
2. 使用指定的 chunk 参数重新分割文档
3. 重新构建向量索引或检索系统

## 文件说明

- **LongCovid.csv**: 包含所有文档的元数据（ID、引用信息、分类等）
- **txt/LongCovid/*.txt**: 8篇核心文献的完整文本内容

这8篇文献是：
1. AAPMRCompendium_NP002 - Multidisciplinary collaborative guidance
2. Al-Aly_39122965 - Long COVID science, research and policy
3. Bateman_34454716 - ME/CFS: Essentials of Diagnosis and Management
4. Fineberg_39110819 - A Long COVID Definition (NASEM)
5. Mueller_40105889 - Long COVID: emerging pathophysiological mechanisms
6. Peluso_39326415 - Mechanisms of long COVID and the path toward therapeutics
7. Vogel_39142505 - Designing and optimizing clinical trials for long COVID
8. Zeraatkar_39603702 - Interventions for the management of long covid

## 注意事项

- 如果只使用默认 chunk 参数（1200/600），系统会使用预构建的 FAISS 索引，不需要这个目录
- 如果使用自定义 chunk 参数，必须有这个目录才能重新构建索引
- CSV 文件包含所有文档的元数据，但只使用了其中8篇核心文献

