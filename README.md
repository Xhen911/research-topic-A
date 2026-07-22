# research-topic-A
f-sum rule 验证双层石墨烯 twist 和 RPA 极化; 谱权重转移在低能子空间有一个特例 (chiral limit: isolated flat band + vanishing AA hopping) 极端平带（态密度无穷大的范霍夫奇点）有跃迁矩阵元限制（与之互相制约的无穷小乘性因子）

## 标杆(1001)
石墨烯吸收谱在可见和近红外波段表现为常数;

## 代码结构（六层架构, v0.2-layered）

```
src/
├── bands/          L1 哈密顿量与本征系统
│   ├── base.py            HamiltonianModel ABC
│   ├── graphene.py        SLG/BLG TB + k·p
│   ├── tbg_bm.py          BM 连续模型
│   ├── tbg_relaxed.py     晶格弛豫 → BM 参数
│   ├── occupations.py     compute_cnp, compute_filling
│   ├── vertices.py        density_form_factor
│   └── symmetry.py        fix_gauge (stub)
├── propagators/    L2 独立粒子传播子
│   ├── lindhard.py        χ₀ 极化 (+ generate_k_mesh)
│   ├── dos.py             DOS / JDOS / A(k,ω)
│   ├── kubo.py            σ(ω) 光电导
│   └── transitions.py     make_bs_cache helper
├── interactions/   L3 相互作用
│   ├── kernel.py          InteractionKernel ABC
│   ├── rpa.py             coulomb_2d + rpa_response
│   └── tddft/bse/hubbard/eph.py   骨架 (NotImplementedError)
├── observables/    L4 可观测量
│   ├── dielectric.py      ε, ε⁻¹, ELF
│   ├── spectral_weight.py compute_swt_1d/2d
│   └── structure_factor.py S(q,ω) = −(1/π) Im χ_RPA
├── validation/     L5 验证层
│   ├── convergence.py     q→0 收敛测试
│   ├── model_checks.py    模型验证套件
│   └── sum_rules/kk/ward/symmetry.py   骨架
├── core/
│   ├── cache.py           CachedModel (k/q 缓存)
│   └── grids.py           make_k_path
└── vis/            fermi_surface 可视化工具
```

依赖方向严格向下：bands ← propagators ← interactions ← observables，validation 横切，core 被各层共用。
回归基线：`TBG_PROD_DATA_DIR=<生产数据目录> python -m pytest tests/ -v`（9 项，全绿才能合并）。