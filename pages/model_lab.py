import time

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import learning_curve, StratifiedKFold

from src.heart_pipeline import (
    MODE_HINTS,
    MODE_LABELS,
    SCREENING_FEATURES,
    CLINICAL_FEATURES,
    XGBClassifier,
    compare_models,
    extract_feature_importance,
    get_param_grid,
    load_model_bundle,
    prepare_xy,
    save_model_bundle,
    tune_model,
)
from config.config import PathConfig


def render_model_lab() -> None:
    st.title("模型训练与评估")
    st.caption("多模型对比 · 参数优化 · 结果验证")
    st.divider()

    data = st.session_state.get("clean_data")
    if data is None:
        st.warning("请先在「数据探索」页面加载并清洗数据。")
        return

    # 模型模式选择
    with st.container(border=True):
        mode = st.selectbox(
            "选择模型模式",
            ["clinical", "screening"],
            format_func=lambda x: MODE_LABELS[x],
            index=0 if st.session_state.get("model_mode") == "clinical" else 1,
            key="model_mode",
        )

        st.info(MODE_HINTS[mode])

    # 检查特征工程页面是否已选择特征
    feat_from_eng = st.session_state.get("selected_features")
    use_feature_eng = False
    if feat_from_eng and len(feat_from_eng) > 0:
        use_feature_eng = st.checkbox(
            "使用「特征工程」页面选中的特征子集",
            value=True,
            help="开启后模型将使用你在特征工程页面筛选的特征。关闭则使用当前模式的默认特征集。",
        )

    if use_feature_eng and feat_from_eng:
        selected_features = [f for f in feat_from_eng if f in data.columns]
        feature_text = f"特征工程自定义选中的 {len(selected_features)} 个特征"
        st.info(f"已启用特征工程选择结果：{', '.join(selected_features)}")
    elif mode == "clinical":
        selected_features = CLINICAL_FEATURES
        feature_text = "包含所有指标（Troponin 和 CK-MB）"
    else:
        selected_features = SCREENING_FEATURES
        feature_text = "剔除 Troponin 和 CK-MB"

    # 过滤数据到所选特征
    available_features = [f for f in selected_features if f in data.columns]
    X, y = prepare_xy(data)
    X = X[available_features]

    st.info(f"当前使用特征：{len(available_features)} 个 — {feature_text}")

    tab1, tab2, tab3 = st.tabs(["多模型对比", "参数优化", "结果验证"])

    # ================================================================
    # TAB 1: 多模型对比
    # ================================================================
    with tab1:
        training_progress_placeholder = st.empty()
        model_progress_placeholder = st.empty()

        # SMOTE 过采样选项
        col_smote, _ = st.columns([1, 3])
        with col_smote:
            use_smote = st.checkbox(
                "启用 SMOTE 过采样",
                value=True,
                help="对训练集进行过采样以平衡正负样本，可提升少数类（阳性）的召回率。需要 imbalanced-learn 库支持。",
            )

        if st.button("开始多模型训练", type="primary", use_container_width=True):
            total_models = len(
                [k for k in ["Logistic Regression", "Decision Tree", "Random Forest",
                              "Gradient Boosting", "SVM", "KNN", "MLP Neural Network", "XGBoost"]]
            )

            with training_progress_placeholder.container():
                st.markdown("**训练进度**")
                progress_bar = st.progress(0)
                status_text = st.empty()

            # 使用 compare_models 的统一训练逻辑，通过 progress_callback 更新 UI
            def _on_progress(name: str, idx: int, total: int) -> None:
                status_text.info(f"正在训练: {name} ({idx}/{total})")
                progress_bar.progress(idx / total)
                with model_progress_placeholder.container():
                    pass  # 实时结果在训练完成后统一展示

            _all_model_count = 8 if XGBClassifier is not None else 7
            # 重新创建进度条以匹配实际模型数
            progress_bar.progress(0)

            comparison = compare_models(
                X, y,
                mode=mode,
                use_smote=use_smote,
                selected_features=available_features,
                progress_callback=_on_progress,
            )

            progress_bar.progress(1.0)
            status_text.success("全部模型训练完成！")

            st.session_state["model_comparison"] = comparison
            st.session_state["best_model"] = comparison["best_model"]
            st.session_state["trained_mode"] = mode

            time.sleep(0.3)
            training_progress_placeholder.empty()
            model_progress_placeholder.empty()

            st.success("模型训练完成！")
            st.rerun()

        comparison = st.session_state.get("model_comparison")
        if comparison is not None:
            results = comparison["results"]

            # 最优模型提示
            best = comparison["best_name"]
            best_f1 = results.iloc[0]["f1"]
            best_auc = results.iloc[0]["auc"]
            st.success(f"**当前最优模型**: {best}  —  F1 = {best_f1:.4f}, AUC = {best_auc:.4f}")

            st.markdown("**模型评估结果**")
            st.dataframe(results, use_container_width=True, hide_index=True)

            # 可视化
            if "training_history" in comparison:
                st.markdown("**训练过程可视化**")
                history_df = pd.DataFrame(comparison["training_history"])

                col_a, col_b = st.columns(2)
                with col_a:
                    fig_acc = px.bar(
                        history_df, x="model", y="accuracy", color="accuracy",
                        color_continuous_scale="Blues", title="Test Accuracy",
                        template="plotly_white",
                    )
                    fig_acc.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30, coloraxis_showscale=False)
                    st.plotly_chart(fig_acc, use_container_width=True)

                with col_b:
                    fig_f1 = px.bar(
                        history_df, x="model", y="f1", color="f1",
                        color_continuous_scale="Purples", title="Test F1 Score",
                        template="plotly_white",
                    )
                    fig_f1.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30, coloraxis_showscale=False)
                    st.plotly_chart(fig_f1, use_container_width=True)

                fig_auc_chart = px.bar(
                    history_df, x="model", y="auc", color="auc",
                    color_continuous_scale="Oranges", title="Test AUC",
                    template="plotly_white",
                )
                fig_auc_chart.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30, coloraxis_showscale=False)
                st.plotly_chart(fig_auc_chart, use_container_width=True)

            # 算法评价分析
            with st.expander("查看各模型详细分析", expanded=False):
                best_accuracy = results["accuracy"].max()
                best_f1 = results["f1"].max()
                best_auc = results["auc"].max()
                avg_accuracy = results["accuracy"].mean()
                avg_f1 = results["f1"].mean()
                avg_auc = results["auc"].mean()

                for _, row in results.iterrows():
                    model_name = row["model"]
                    acc_gap = best_accuracy - row["accuracy"]
                    auc_gap = best_auc - row["auc"]
                    metrics_std = results[results["model"] == model_name][["accuracy", "f1", "auc"]].std(axis=1).values[0]

                    if "cv_f1_mean" in row and pd.notna(row["cv_f1_mean"]):
                        overfit_gap = row["cv_f1_mean"] - row["f1"]
                    else:
                        overfit_gap = 0

                    score = (row["accuracy"] + row["f1"] + row["auc"]) / 3
                    grade = "A" if score >= 0.85 else "B" if score >= 0.75 else "C" if score >= 0.65 else "D"

                    with st.container(border=True):
                        st.markdown(f"**{model_name}** — 综合评级: **{grade}**")
                        metrics_text = (
                            f"准确率 {row['accuracy']:.4f} | F1 {row['f1']:.4f} | AUC {row['auc']:.4f}  |  "
                            f"均衡性 σ={metrics_std:.4f}  |  过拟合差 {overfit_gap:.4f}"
                        )
                        st.caption(metrics_text)

                        suggestions = []
                        if auc_gap > 0.05 and row["auc"] < avg_auc:
                            suggestions.append("AUC相对较低，可尝试调整分类阈值")
                        if acc_gap > 0.05 and row["accuracy"] < avg_accuracy:
                            suggestions.append("准确率低于平均，考虑增加正则化")
                        if overfit_gap > 0.05:
                            suggestions.append("存在过拟合，建议增强正则化或增加数据")
                        if metrics_std > 0.05:
                            suggestions.append("指标不均衡，需针对性调优各指标")
                        if not suggestions:
                            suggestions.append("该算法在本项目中表现良好，可直接使用")

                        for s in suggestions:
                            st.markdown(f"- {s}")

            # 项目整体总结
            st.info(
                f"本次训练共 {len(results)} 个模型参与对比。"
                f"平均准确率 **{avg_accuracy:.4f}**，平均 F1 **{avg_f1:.4f}**，平均 AUC **{avg_auc:.4f}**。"
                f"最佳模型为 **{results.iloc[0]['model']}**。"
            )

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("**模型指标对比**")
                fig = px.bar(
                    results.melt(id_vars="model", value_vars=["accuracy", "f1", "auc"]),
                    x="model", y="value", color="variable", barmode="group", template="plotly_white",
                )
                fig.update_layout(plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-30, legend_title_text="指标")
                st.plotly_chart(fig, use_container_width=True)

            with col_right:
                st.markdown("**ROC 曲线**")
                roc_fig = go.Figure()
                for name, curve in comparison["roc_curves"].items():
                    roc_fig.add_trace(go.Scatter(x=curve["fpr"], y=curve["tpr"], mode="lines", name=name))
                roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash")))
                roc_fig.update_layout(
                    xaxis_title="FPR", yaxis_title="TPR", plot_bgcolor="white", paper_bgcolor="white", legend_title_text="模型",
                )
                st.plotly_chart(roc_fig, use_container_width=True)

        else:
            st.info("点击上方按钮开始训练多个模型进行对比。支持 8 种机器学习模型。")

    # ================================================================
    # TAB 2: 参数优化
    # ================================================================
    with tab2:
        with st.container(border=True):
            comparison = st.session_state.get("model_comparison")
            default_model = comparison["best_name"] if comparison is not None else "Random Forest"

            all_model_options = [
                "Logistic Regression",
                "Decision Tree",
                "Random Forest",
                "Gradient Boosting",
                "SVM",
                "KNN",
                "MLP Neural Network",
            ]
            if XGBClassifier is not None:
                all_model_options.append("XGBoost")

            model_name = st.selectbox(
                "选择待优化模型",
                options=all_model_options,
                index=(all_model_options.index(default_model) if default_model in all_model_options else 2),
            )

            use_smote_tune = st.checkbox("启用 SMOTE 过采样（调优时）", value=True, key="use_smote_tune")

            if XGBClassifier is None and model_name == "XGBoost":
                st.warning("当前环境未安装 XGBoost，该选项不可用。")
            elif st.button("执行参数优化", use_container_width=True):
                tuning_progress = st.empty()
                tuning_status = st.empty()

                with tuning_progress.container():
                    st.markdown("**参数优化进度**")
                    tuning_bar = st.progress(0)

                param_grid = get_param_grid(model_name, fast_mode=True)
                total_combinations = 1
                for v in param_grid.values():
                    total_combinations *= len(v)

                tuning_status.info(f"正在优化: {model_name} — 共 {total_combinations} 种参数组合")

                with st.spinner("网格搜索中..."):
                    try:
                        tuning = tune_model(X, y, model_name, mode=mode, use_smote=use_smote_tune)
                    except Exception as e:
                        tuning_progress.empty()
                        tuning_status.empty()
                        st.error(f"参数优化失败：{e}")
                        st.info("提示：电脑内存不足时可以关闭其他程序再试，或者选择更简单的模型（如 Logistic Regression）进行优化。")
                        return

                tuning_bar.progress(1.0)
                tuning_status.success("参数优化完成！")

                time.sleep(0.5)
                tuning_progress.empty()
                tuning_status.empty()

                st.session_state["best_model"] = tuning["best_model"]
                st.session_state["tuning_result"] = tuning

                st.success("参数优化完成！")
                st.markdown("**最佳参数**")
                st.json(tuning["best_params"])

                st.markdown("**参数优化可视化**")
                cv_results = tuning["cv_results"]

                if len(cv_results) > 1:
                    numeric_params = []
                    for col in cv_results.columns:
                        if col.startswith("param_model__") and cv_results[col].dtype in ["int64", "float64"]:
                            numeric_params.append(col)

                    if numeric_params:
                        viz_a, viz_b = st.columns(2)
                        param_col = numeric_params[0]
                        cv_results_viz = cv_results.copy()
                        cv_results_viz["param_value"] = cv_results_viz[param_col].apply(lambda x: float(x) if x is not None else 0)

                        with viz_a:
                            fig_params = px.scatter(
                                cv_results_viz, x="param_value", y="mean_test_score", error_y="std_test_score",
                                title=f"{param_col.replace('param_model__', '')} vs F1 Score",
                                template="plotly_white",
                                labels={"param_value": param_col.replace("param_model__", ""), "mean_test_score": "F1 Score"},
                            )
                            fig_params.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                            st.plotly_chart(fig_params, use_container_width=True)

                        with viz_b:
                            cv_viz_data = cv_results[["mean_test_score", "std_test_score", "rank_test_score"]].head(10).copy()
                            cv_viz_data["param组合"] = [f"组合{i+1}" for i in range(len(cv_viz_data))]
                            fig_cv = px.bar(
                                cv_viz_data, x="param组合", y="mean_test_score", error_y="std_test_score",
                                color="mean_test_score", color_continuous_scale="Viridis",
                                title="Top 10 Parameter Combinations", template="plotly_white",
                            )
                            fig_cv.update_layout(
                                plot_bgcolor="white", paper_bgcolor="white", xaxis_tickangle=-45, coloraxis_showscale=False,
                            )
                            st.plotly_chart(fig_cv, use_container_width=True)

                st.markdown("**测试集指标**")
                st.dataframe(pd.DataFrame([tuning["metrics"]]), use_container_width=True, hide_index=True)

                st.markdown("**交叉验证结果（Top 10）**")
                st.dataframe(
                    tuning["cv_results"][["params", "mean_test_score", "std_test_score", "rank_test_score"]].head(10),
                    use_container_width=True,
                    hide_index=True,
                )

    # ================================================================
    # TAB 3: 结果验证
    # ================================================================
    with tab3:
        with st.container(border=True):
            model = st.session_state.get("best_model")
            comparison = st.session_state.get("model_comparison")
            if model is None or comparison is None:
                st.info("请先完成模型训练")
                return

            trained_mode = comparison.get("trained_mode", "clinical")
            if trained_mode != mode:
                st.warning(
                    f"当前模型是使用「{'临床版' if trained_mode == 'clinical' else '筛查版'}」模式训练的，"
                    f"请重新训练模型以匹配当前「{'临床版' if mode == 'clinical' else '筛查版'}」模式。"
                )
                return

            X_test = comparison["X_test"]
            y_test = comparison["y_test"]
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]
            cm = confusion_matrix(y_test, y_pred)

            st.markdown("**模型验证结果**")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**混淆矩阵**")
                cm_fig = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Negative", "Positive"], y=["Negative", "Positive"], template="plotly_white",
                )
                cm_fig.update_layout(plot_bgcolor="white", paper_bgcolor="white")
                st.plotly_chart(cm_fig, use_container_width=True)

            with col_b:
                st.markdown("**特征重要性**")
                importance = extract_feature_importance(model, data)
                if not importance.empty:
                    fig = px.bar(
                        importance.head(12), x="importance", y="feature", orientation="h",
                        color="importance", color_continuous_scale="Purples", template="plotly_white",
                    )
                    fig.update_layout(yaxis=dict(autorange="reversed"), plot_bgcolor="white", paper_bgcolor="white", coloraxis_showscale=False)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.caption("当前最优模型不直接提供特征重要性")

            # ── 学习曲线 ────────────────────────────────────────────────────────
            st.markdown("**学习曲线**")
            st.caption("展示训练集与交叉验证集得分随训练样本量的变化趋势，用于判断过拟合/欠拟合。")

            with st.spinner("计算学习曲线中..."):
                X_learn = pd.concat([comparison["X_train"], comparison["X_test"]])
                y_learn = pd.concat([comparison["y_train"], comparison["y_test"]])

                train_sizes, train_scores, valid_scores = learning_curve(
                    clone(comparison["best_model"]),
                    X_learn, y_learn,
                    train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring="f1",
                    n_jobs=1,
                )

                train_mean = train_scores.mean(axis=1)
                train_std = train_scores.std(axis=1)
                valid_mean = valid_scores.mean(axis=1)
                valid_std = valid_scores.std(axis=1)

                lc_fig = go.Figure()
                lc_fig.add_trace(go.Scatter(
                    x=train_sizes * len(X_learn),
                    y=train_mean,
                    mode="lines+markers",
                    name="训练集得分",
                    line=dict(color="#3b82f6", width=2),
                    marker=dict(size=6),
                    error_y=dict(type="data", array=train_std, visible=True, color="#93c5fd", thickness=1),
                ))
                lc_fig.add_trace(go.Scatter(
                    x=train_sizes * len(X_learn),
                    y=valid_mean,
                    mode="lines+markers",
                    name="交叉验证得分",
                    line=dict(color="#ef4444", width=2),
                    marker=dict(size=6),
                    error_y=dict(type="data", array=valid_std, visible=True, color="#fca5a5", thickness=1),
                ))
                lc_fig.add_hline(
                    y=valid_mean[-1],
                    line_dash="dash",
                    line_color="#94a3b8",
                    annotation_text=f"最终 CV F1 = {valid_mean[-1]:.3f}",
                    annotation_font=dict(size=11, color="#64748b"),
                )
                lc_fig.update_layout(
                    xaxis_title="训练样本数",
                    yaxis_title="F1 得分",
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    font=dict(size=12, color="#4a5568"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=350,
                )
                lc_fig.update_xaxes(gridcolor="#edf2f7")
                lc_fig.update_yaxes(gridcolor="#edf2f7", range=[0, 1])
                st.plotly_chart(lc_fig, use_container_width=True)

                # 过拟合判断
                gap = train_mean[-1] - valid_mean[-1]
                if gap > 0.15:
                    st.error(f"**过拟合风险较高**：训练集与验证集得分差距 {gap:.3f}，建议增加正则化或扩大数据量。")
                elif gap > 0.08:
                    st.warning(f"**轻微过拟合**：训练集与验证集得分差距 {gap:.3f}，可尝试降低模型复杂度。")
                elif gap > 0.03:
                    st.info(f"**拟合状态尚可**：训练集与验证集得分差距 {gap:.3f}。")
                else:
                    st.success(f"**拟合状态良好**：训练集与验证集得分差距 {gap:.3f}，模型泛化能力较强。")

            col_save, col_load = st.columns(2)
            with col_save:
                if st.button("保存当前最优模型", use_container_width=True):
                    path = save_model_bundle(model)
                    st.success(f"模型已保存到: {path}")
            with col_load:
                saved_path = PathConfig.MODEL_DIR / "best_model.joblib"
                if saved_path.exists():
                    if st.button("加载已保存模型", use_container_width=True):
                        try:
                            bundle = load_model_bundle(saved_path)
                            st.session_state["best_model"] = bundle["model"]
                            st.session_state["last_loaded_model"] = True
                            st.success(f"已加载模型: {saved_path.name}")
                            st.rerun()
                        except Exception as e:
                            st.error(f"加载失败: {e}")
                else:
                    st.caption("暂无已保存的模型")

            st.info("**MLP Neural Network** 作为深度学习部分，用于和传统机器学习模型做性能对比。")
