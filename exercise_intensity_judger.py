try:
    age = float(input("请输入你的年龄："))
    heart_rate = float(input("请输入运动时的平均心率："))
except ValueError:
    print("输入的数据不太合理，请输入数字。")
else:
    if age <= 0 or age > 120 or heart_rate <= 0 or heart_rate > 230:
        print("输入的数据不太合理，请检查年龄和心率。")
    else:
        max_heart_rate = 208 - 0.7 * age
        rate = heart_rate / max_heart_rate * 100

        if rate < 50:
            level = "轻度运动"
            advice = "可以适当增加运动时间或强度。"
        elif rate < 70:
            level = "中等强度运动"
            advice = "这个强度比较合适，可以继续保持。"
        elif rate < 90:
            level = "高强度运动"
            advice = "运动强度较大，要注意休息和补水。"
        else:
            level = "强度过高"
            advice = "建议放慢速度，避免身体负担过重。"

        print(f"估算最大心率约为 {max_heart_rate:.1f} 次/分钟。")
        print(f"本次运动强度约为 {rate:.1f}%。")
        print(f"判断结果：你的运动属于{level}，{advice}")
