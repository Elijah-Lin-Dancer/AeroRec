    # w_base=0.0：候选池已含热度信息，混排阶段不再二次注入热门偏差（见 mixer 文档）
    mixer = Mixer(w_base=0.0, w_ctr=1.0, max_same_category=2, top_k=TOP_K)