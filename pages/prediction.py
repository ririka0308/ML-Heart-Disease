import datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.heart_pipeline import MODE_LABELS, MODE_HINTS, RISK_COLORS, build_single_case, predict_risk
from src.pdf_generator import create_pdf_report_for_prediction


def render_prediction() -> None:
    st.title("个体风险预测")
    st.caption("输入患者指标，预测心脏病患病风险")
    st.divider()

    model = st.session_state.get("best_model")
    data = st.session_state.get("clean_data")
    trained_mode = st.session_state.get("trained_mode", "clinical")

    if model is None or data is None:
        st.warning("请先完成数据清洗和模型训练。")
        return

    with st.container(border=True):
        mode = st.selectbox(
            "选择预测模式",
            ["clinical", "screening"],
            format_func=lambda x: MODE_LABELS[x],
            index=0 if trained_mode == "clinical" else 1,
            key="prediction_mode"
        )
        if mode != trained_mode:
            st.warning(
                f"模式不匹配：训练的模型是「{MODE_LABELS[trained_mode]}」，但选择了「{MODE_LABELS[mode]}」。请在模型训练页重新训练。"
            )
        st.info(MODE_HINTS[mode])

    defaults = build_single_case(data.median(numeric_only=True))

    with st.container(border=True):
        st.markdown("**患者信息输入**")
        st.caption("输入单个患者的基础指标，系统将输出心脏病风险概率和分类结果。")

        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("性别", ["Male", "Female"], index=0 if defaults["Gender"] == "Male" else 1)
        with c2:
            age = st.number_input("年龄", min_value=1, max_value=120, value=int(defaults["Age"]))
        with c3:
            heart_rate = st.number_input("心率", min_value=20, max_value=220, value=int(defaults["Heart rate"]))

        c4, c5, c6 = st.columns(3)
        with c4:
            sbp = st.number_input("收缩压", min_value=60, max_value=250, value=int(defaults["Systolic blood pressure"]))
        with c5:
            dbp = st.number_input("舒张压", min_value=30, max_value=180, value=int(defaults["Diastolic blood pressure"]))
        with c6:
            sugar = st.number_input("血糖", min_value=20.0, max_value=600.0, value=float(defaults["Blood sugar"]), step=1.0)

        ckmb = float(defaults["CK-MB"])
        troponin = float(defaults["Troponin"])
        if mode == "clinical":
            c7, c8 = st.columns(2)
            with c7:
                ckmb = st.number_input("CK-MB", min_value=0.0, max_value=300.0, value=float(defaults["CK-MB"]), step=0.1)
            with c8:
                troponin = st.number_input("肌钙蛋白", min_value=0.0, max_value=12.0, value=float(defaults["Troponin"]), step=0.001, format="%.3f")

    case = {
        "Gender": gender,
        "Age": age,
        "Heart rate": heart_rate,
        "Systolic blood pressure": sbp,
        "Diastolic blood pressure": dbp,
        "Blood sugar": sugar,
        "CK-MB": ckmb,
        "Troponin": troponin,
    }

    if st.button("执行风险预测", type="primary", use_container_width=True):
        with st.spinner("正在分析患者数据..."):
            result = predict_risk(model, case, mode=mode)
        st.session_state["last_prediction"] = result
        st.session_state["last_case"] = case
        st.session_state["corrected_case"] = result.get("corrected_case", case)

        if "prediction_history" not in st.session_state:
            st.session_state["prediction_history"] = []

        history_item = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "case": case,
            "result": result,
            "mode": mode
        }
        st.session_state["prediction_history"].append(history_item)
        st.success("预测完成！")

    result = st.session_state.get("last_prediction")
    if result is not None:
        with st.container(border=True):
            st.markdown("**预测结果**")
            col1, col2, col3 = st.columns(3)

            pred_text = "阳性" if result["prediction"] == "positive" else "阴性"
            pred_color = RISK_COLORS["高风险"] if result["prediction"] == "positive" else RISK_COLORS["低风险"]
            with col1:
                st.markdown(
                    f'<div style="background: #fef2f2; border-radius: 10px; padding: 1rem; text-align: center;">'
                    f'<div style="color: #9ca3af; font-size: 0.85rem;">预测结果</div>'
                    f'<div style="color: {pred_color}; font-size: 1.6rem; font-weight: 700;">{pred_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            prob_color = (
                RISK_COLORS["高风险"]
                if result["probability"] >= 0.7
                else RISK_COLORS.get(result["risk_level"], RISK_COLORS["低风险"])
            )
            with col2:
                st.markdown(
                    f'<div style="background: #f0fdf4; border-radius: 10px; padding: 1rem; text-align: center;">'
                    f'<div style="color: #9ca3af; font-size: 0.85rem;">患病概率</div>'
                    f'<div style="color: {prob_color}; font-size: 1.6rem; font-weight: 700;">{result["probability"] * 100:.2f}%</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            with col3:
                st.markdown(
                    f'<div style="background: #fefce8; border-radius: 10px; padding: 1rem; text-align: center;">'
                    f'<div style="color: #9ca3af; font-size: 0.85rem;">风险等级</div>'
                    f'<div style="color: {RISK_COLORS.get(result["risk_level"], "#333")}; font-size: 1.6rem; font-weight: 700;">{result["risk_level"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**预测概率分布**")
            chart_df = pd.DataFrame(
                {"category": ["阴性", "阳性"], "probability": [1 - result["probability"], result["probability"]]}
            )
            fig = go.Figure(go.Bar(
                x=chart_df["category"],
                y=chart_df["probability"],
                marker_color=["#3b82f6", "#ef4444"],
                text=[f"{(1 - result['probability']) * 100:.1f}%", f"{result['probability'] * 100:.1f}%"],
                textposition="auto",
            ))
            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                yaxis_range=[0, 1], yaxis_title="概率", xaxis_title="诊断结果",
                font={"color": "#64748b"},
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("**健康建议**")
            risk_level = result["risk_level"]
            suggestions = []
            if mode == "screening":
                suggestions.append("*筛查模式提示*：当前无心肌损伤生化指标，建议进一步做临床检查")
                if risk_level == "高风险":
                    suggestions.append("建议立即就医并进行肌钙蛋白/CK-MB检查")
                elif risk_level == "中风险":
                    suggestions.append("建议2-3周内复查，注意监测血压血糖")
                else:
                    suggestions.append("保持健康生活方式，定期体检")
            else:
                if risk_level == "高风险":
                    suggestions.append("请携带报告咨询心血管专科医生")
                elif risk_level == "中风险":
                    suggestions.append("建议改善生活方式，1个月后随访")
                else:
                    suggestions.append("继续保持健康生活方式")

            if result.get("warnings"):
                for warn in result["warnings"]:
                    st.warning(warn)
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")

            st.markdown("**输入参数与参考值对照**")
            ref_data = [["指标", "检测值", "单位", "参考范围", "状态"]]
            ref_ranges = {
                'Age': ('年龄', '岁', '18-65'),
                'Heart rate': ('心率', 'bpm', '60-100'),
                'Systolic blood pressure': ('收缩压', 'mmHg', '<120'),
                'Diastolic blood pressure': ('舒张压', 'mmHg', '<80'),
                'Blood sugar': ('血糖', 'mg/dL', '<100'),
                'CK-MB': ('CK-MB', 'ng/mL', '<2.5'),
                'Troponin': ('肌钙蛋白', 'ng/mL', '<0.04'),
                'Gender': ('性别', '', '-'),
            }
            hidden_markers = ["CK-MB", "Troponin"] if mode == "screening" else []
            case_data = st.session_state["last_case"]
            for key, value in case_data.items():
                if key in hidden_markers:
                    continue
                if key in ref_ranges:
                    name, unit, ref = ref_ranges[key]
                    status = "正常"
                    try:
                        val = float(value)
                        if key == "Systolic blood pressure":
                            if val >= 140: status = "偏高"
                            elif val >= 120: status = "临界"
                        elif key == "Diastolic blood pressure":
                            if val >= 90: status = "偏高"
                            elif val >= 80: status = "临界"
                        elif key == "Blood sugar":
                            if val >= 126: status = "偏高"
                            elif val >= 100: status = "临界"
                        elif key == "CK-MB":
                            if val >= 6.0: status = "偏高"
                            elif val >= 2.5: status = "临界"
                        elif key == "Troponin":
                            if val >= 0.4: status = "偏高"
                            elif val >= 0.04: status = "临界"
                        elif key == "Heart rate":
                            if val < 50 or val > 120: status = "异常"
                            elif val < 60 or val > 100: status = "临界"
                    except Exception:
                        pass
                    ref_data.append([name, str(value), unit, ref, status])

            ref_df = pd.DataFrame(ref_data[1:], columns=ref_data[0])
            st.dataframe(ref_df, use_container_width=True, hide_index=True)
            st.caption("正常 = 指标在健康范围  临界 = 需关注  偏高/异常 = 建议就医")

            st.markdown("**生成诊断报告**")
            if st.button("生成 PDF 报告", use_container_width=True):
                with st.spinner("正在生成报告..."):
                    pdf_bytes = create_pdf_report_for_prediction(case_data, result, mode=mode)
                st.session_state["pdf_bytes"] = pdf_bytes
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                st.session_state["pdf_filename"] = f"心脏病风险预测报告_{timestamp}.pdf"
                st.rerun()
            pdf_bytes = st.session_state.get("pdf_bytes")
            if pdf_bytes is not None:
                st.download_button(
                    label="点击下载 PDF",
                    data=pdf_bytes,
                    file_name=st.session_state.get("pdf_filename", "report.pdf"),
                    mime="application/pdf",
                    use_container_width=True,
                )

        st.warning(
            "**免责声明**：该结果用于毕业设计中的临床辅助决策演示，不应替代真实医生诊断。如有健康疑虑，请咨询专业医疗人员。"
        )

    if st.session_state.pop("history_cleared", False):
        st.success("历史记录已清空")

    prediction_history = st.session_state.get("prediction_history", [])
    if prediction_history:
        with st.container(border=True):
            st.markdown("**预测历史**")

            col_export, col_clear = st.columns([1, 1])
            with col_export:
                if st.button("批量导出为 CSV", use_container_width=True):
                    history_data = []
                    for item in prediction_history:
                        row = {
                            "时间": item["timestamp"],
                            "模式": "临床版" if item["mode"] == "clinical" else "筛查版",
                            "性别": item["case"]["Gender"],
                            "年龄": item["case"]["Age"],
                            "心率": item["case"]["Heart rate"],
                            "收缩压": item["case"]["Systolic blood pressure"],
                            "舒张压": item["case"]["Diastolic blood pressure"],
                            "血糖": item["case"]["Blood sugar"],
                            "预测结果": "阳性" if item["result"]["prediction"] == "positive" else "阴性",
                            "患病概率": f"{item['result']['probability'] * 100:.2f}%",
                            "风险等级": item["result"]["risk_level"],
                        }
                        if item["mode"] == "clinical":
                            row["CK-MB"] = item["case"]["CK-MB"]
                            row["肌钙蛋白"] = item["case"]["Troponin"]
                        history_data.append(row)
                    history_df = pd.DataFrame(history_data)
                    csv_data = history_df.to_csv(index=False, encoding="utf-8-sig")
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    st.download_button(
                        label="下载 CSV",
                        data=csv_data,
                        file_name=f"预测历史_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
            with col_clear:
                if st.button("清空历史记录", use_container_width=True):
                    st.session_state["prediction_history"] = []
                    st.session_state["history_cleared"] = True
                    st.rerun()

            display_history = []
            for i, item in enumerate(reversed(prediction_history), 1):
                display_history.append({
                    "序号": i,
                    "时间": item["timestamp"],
                    "性别": item["case"]["Gender"],
                    "年龄": item["case"]["Age"],
                    "预测结果": "阳性" if item["result"]["prediction"] == "positive" else "阴性",
                    "概率": f"{item['result']['probability'] * 100:.1f}%",
                    "风险等级": item["result"]["risk_level"],
                })
            history_df = pd.DataFrame(display_history)
            st.dataframe(history_df, use_container_width=True, hide_index=True)
