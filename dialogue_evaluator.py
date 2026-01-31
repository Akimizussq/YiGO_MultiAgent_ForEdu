"""
对话评价模块
基于大模型的对话行为分类评价
"""

import openai
import os
import json
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv()

# 定义分类层级
CLASS_HIERARCHY = {
    "off_topic": 1,      # 非话题性
    "questioning": 2,    # 提问
    "stating": 3,        # 陈述
    "supporting": 4,     # 支持
    "challenging": 5,    # 冲突
    "clarifying": 6,     # 澄清
    "summarizing": 7,    # 总结
    "evaluating": 8,     # 评价
}


class DialogueEvaluator:
    """对话行为评价器"""

    def __init__(self):
        """初始化评价器"""
        self.client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL")
        )
        self.model = os.getenv("OPENAI_MODEL_NAME", "gpt-4")
        self.instructions = self._build_evaluator_instructions()

    def _build_evaluator_instructions(self) -> str:
        """构建评价器的指令"""
        category_descriptions = {
            # 基础层级
            "off_topic": "非话题性:与讨论主题无关的内容,包括寒暄、情绪表达、无意义回复等。",

            "questioning": "提问:提出问题以获取信息、澄清概念、或引导他人解释。必须包含疑问词或问号。",

            "stating": "陈述:提供事实、背景、经验或中性描述,不包含对他人观点的详细解释（如果有，需要归类为澄清），不包含推理、判断或明确态度立场。",

            # 中间层级
            "supporting": """支持:明确表示同意某观点。
            - 简单支持:仅表达赞同(如"我同意"、"赞成")
            - 有据支持:表达赞同+提供理由/证据/案例(如"我同意,因为..."、"赞同,例如...")
            【识别重点】需要表示同意态度,且具有充足的论据（至少包含1-2个具体理由或案例），否则归类为"陈述" """,

            "challenging": """冲突:表达不同意见、反驳或提出反对观点。
            - 简单反对:仅表达否定(如"我不同意"、"这不对")
            - 有据反对:表达反对+提供反驳理由/反例/逻辑分析(如"不同意,因为..."、"但实际上...")
            【识别重点】需要表示反对态度,且具有充足的论据（至少包含1-2个具体反驳点），否则归类为"陈述" """,

            "clarifying": """澄清:对某观点进行识别、解释、补充细节、或消除歧义，使信息更清晰具象。
            - 简单澄清:简单定义或重述(如"XX是指...")
            - 结构化澄清:通过举例/类比/引用他人观点来详细说明(如"举例来说..."、"你提到的XX其实是...")
            【识别重点】关注是否在"解释说明"而非"表态"，有无实质信息增量。如果只是列举事实而没有解释说明，应归类为"陈述" """,

            # 高层级
            "summarizing": "总结:对讨论内容进行概括、提炼、结构化总结,不加入价值判断。呈现讨论全貌或框架。简单的发言一般不算作总结（而算作陈述），总结需要有完整的逻辑。",

            "evaluating": "评价:基于讨论内容进行价值判断、优先级排序、或给出决策倾向。包含'应该'、'建议'等判断性词汇。简单的发言一般不算作评价（而算作陈述），评价需要有完整的逻辑。"
        }

        prompt = """
            【核心判定原则 - 主要意图优先】

            当文本包含多种意图时，请按照以下优先级顺序判定：
            1. 识别文本的**核心目的**和**主要观点**
            2. 确定作者想要传达的**主要态度**（支持/反对/中立）
            3. 如果表态成分占主导（>60%），归为支持/冲突/澄清
            4. 如果信息提供成分占主导（>60%），归为陈述/提问
            5. 如果综合归纳成分占主导（>60%），归为总结/评价

            【判定示例】
            - "我同意你的观点，但想补充一点背景信息..." → 主要意图是支持，归类为 supporting
            - "这个观点有问题，因为..." + 大量反驳论证 → 主要意图是反对，归类为 challenging
            - "你提到的概念其实是..." + 详细解释说明 → 主要意图是解释，归类为 clarifying

            你是一个专业的小组话题讨论活动的文本数据处理专家。请根据下面提供的分类体系，将用户输入的文本划分到最合适的类别中。注意，你需要对每一个发言都确定其分类(在以下分类类型中判断并选取)，不可以不进行分类。\n
            【注意】你的回答必须是一个JSON对象，格式如下：{"classification": "类别名称"}\n\n
            请你对于发言文本，了解完整段文本的逻辑后再进行评价，而不是根据某个关键词捕捉就进行武断评价，而忽视了后面的内容。
            --- 分类体系 ---\n
        """

        for key, desc in category_descriptions.items():
            prompt += f"- {key}: {desc}\\n"

        prompt += """
            -----------------------------------
            【Few-shot 示例】
            -----------------------------------

            （低层级示例）

            输入：
            "目前，主流的AI教育产品主要集中在自适应学习和智能测评两个领域。例如，一些平台会根据学生的答题情况，动态调整后续的练习难度。"
            输出：
            {"classification": "stating"}

            输入：
            "我想了解一下，在引入这些智能技术后，教师的角色具体会发生哪些变化？他们是需要更多地介入指导，还是可以把更多精力放在课程设计上？"
            输出：
            {"classification": "questioning"}

            输入: "大家好,很高兴参与讨论。/ 没有提交观点的同学请在规定时间内提交"
            输出: {"classification": "off_topic"}

            -----------------------------------

            （中间层级示例）

            ## supporting 示例 （对观点表支持）

            输入: "我非常同意刚才A同学的观点。AI确实能极大地解放教师的生产力，让他们从重复性的批改工作中解脱出来，我们学校上学期引进了类似的系统，效果非常显著。"
            输出: {"classification": "supporting"}

            输入: "我非常赞同你的观点。根据教育部2023年发布的报告，采用AI辅助教学的学校，学生平均成绩提升了15%，教师备课时间减少了40%。这些数据充分证明了AI技术在教育领域的应用价值。"
            输出: {"classification": "supporting"}

            输入: "完全认同你刚才提到的看法。我在实际教学中也观察到了同样的现象，使用智能批改系统后，学生反馈更加及时，学习积极性明显提高。特别是在作文批改方面，效果尤为显著。"
            输出: {"classification": "supporting"}

            输入: "支持这个方案，我认为它抓住了教育信息化的关键。首先，它解决了个性化学习的难题；其次，通过数据分析可以精准定位学生的薄弱环节；最后，这种模式可以大规模推广，成本效益很高。"
            输出: {"classification": "supporting"}

            ---

            ## challenging 示例 （对观点表反对）

            输入: "这个想法可能需要更谨慎地考虑。虽然技术上可行，但完全依赖AI进行评价可能会忽略学生的创新思维和批判性思维，这些是目前算法很难衡量的。"
            输出: {"classification": "challenging"}

            输入: "我不同意你的看法。你的论证存在一个关键问题：你假设AI系统能够完全理解复杂的教育场景，但实际上目前的技术水平还远未达到。教育不仅仅是知识传递，还涉及情感交流、价值观培养等AI难以处理的维度。"
            输出: {"classification": "challenging"}

            输入: "这个观点需要商榷。根据最新的研究数据，完全依赖AI进行个性化学习的学生，其批判性思维能力反而下降了约20%。这说明过度依赖智能技术可能会削弱学生的独立思考能力。"
            输出: {"classification": "challenging"}

            输入: "虽然这个方案在理论上可行，但在实际应用中会遇到很多限制。首先，不是所有学校都有足够的资金投入；其次，农村地区的网络基础设施难以支撑；最后，教师的技术素养参差不齐，培训成本很高。"
            输出: {"classification": "challenging"}

            输入: "我认为这个方案不是最优选择。与其完全依赖AI系统，不如采用混合模式：AI负责基础的知识点练习和测评，而教师专注于高阶思维培养和情感关怀。这样既能发挥技术优势，又能保留教育的本质。"
            输出: {"classification": "challenging"}

            ---

            ## clarifying 示例 (解释说明,可简可详)

            输入: "我来补充一下刚才提到的'过程性评价'。它不仅仅是记录对错，更重要的是分析学生的解题路径和思维过程，比如他们在哪一步卡住了，或者用了哪种不同的解法。"
            输出: {"classification": "clarifying"}

            输入: "我想澄清一下'自适应学习'和'个性化学习'的区别。自适应学习是根据学生答题情况实时调整题目难度，是个性化学习的一种技术实现；而个性化学习更强调根据学生整体学习风格和需求定制教学方案，范围更广。"
            输出: {"classification": "clarifying"}

            输入: "补充一些背景信息，帮助大家更好地理解这个问题。这个技术最早起源于20世纪80年代的智能导学系统，但受限于当时的计算能力，应用范围很小。随着深度学习的发展，近年来才真正实现了大规模应用。"
            输出: {"classification": "clarifying"}

            输入: "我来详细拆解一下这个评价体系。它包含三个核心模块：第一个是知识掌握度评估，通过答题准确率来衡量；第二个是学习行为分析，关注学习时长和频率；第三个是能力发展追踪，记录批判性思维和创新能力的变化。"
            输出: {"classification": "clarifying"}

            -----------------------------------

            （负面示例 - 容易混淆的类别）

            输入: "研究表明，AI教育应用可以提高学习效率约30%。"
            输出: {"classification": "stating"}
            说明: 【负面示例】这是中性事实陈述，没有表达赞同态度，不应归类为supporting

            输入: "但是，AI系统在处理复杂推理任务时准确率只有65%。"
            输出: {"classification": "stating"}
            说明: 【负面示例】这是中性事实陈述，没有明确表达反对态度，不应归类为challenging

            输入: "过程性评价包括答题记录、学习路径、思维过程等多个方面。"
            输出: {"classification": "stating"}
            说明: 【负面示例】这是列举事实，没有对已有观点进行解释说明，不应归类为clarifying

            输入: "我完全同意你的看法，这个方案确实很有价值。"
            输出: {"classification": "supporting"}
            说明: 【负面示例】这是明确表态支持，没有解释说明的意图，不应归类为clarifying

            输入: "虽然你提到的方案有优势，但也存在一些实施成本过高的问题。"
            输出: {"classification": "challenging"}
            说明: 【负面示例】虽然先承认优势，但核心意图是指出问题，不应归类为supporting

            -----------------------------------

            （高层级示例）

            输入：
            "总结一下，智能技术在教育测评中有三大应用方向：多模态数据采集、过程性多元评价、个性化学习路径定制。"
            输出：
            {"classification": "summarizing"}

            输入：
            "综合评估，我们应优先解决AI教育应用的隐私合规问题，因为这是落地的核心前提。"
            输出：
            {"classification": "evaluating"}


        """

        prompt += "\\n请严格按照上述规则对用户发送的文本进行分类。"

        return prompt

    def evaluate(self, text: str) -> str:
        """
        评价文本的对话行为

        Args:
            text: 待评价的文本

        Returns:
            分类结果（分类名称）
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self.instructions
                    },
                    {
                        "role": "user",
                        "content": text
                    }
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )

            result = response.choices[0].message.content
            result_json = json.loads(result)
            classification = result_json.get("classification", "stating")

            # 验证分类是否有效
            if classification not in CLASS_HIERARCHY:
                print(f"[警告] 无效分类: {classification}，默认归类为 stating")
                classification = "stating"

            return classification

        except Exception as e:
            print(f"[错误] 评价失败: {e}")
            return "stating"  # 默认返回陈述

    def evaluate_batch(self, texts: List[str]) -> List[str]:
        """
        批量评价文本

        Args:
            texts: 待评价的文本列表

        Returns:
            分类结果列表
        """
        results = []
        for text in texts:
            result = self.evaluate(text)
            results.append(result)
        return results


class DialogueStatistics:
    """对话统计分析"""

    def __init__(self):
        """初始化统计数据"""
        self.data = {
            "角色": [],
            "off_topic": [],
            "questioning": [],
            "stating": [],
            "supporting": [],
            "challenging": [],
            "clarifying": [],
            "summarizing": [],
            "evaluating": []
        }
        self.role_names = []
        self.class_names = list(CLASS_HIERARCHY.keys())

    def add_speech(self, role: str, classification: str):
        """
        添加一条发言记录

        Args:
            role: 角色名称
            classification: 分类结果
        """
        if role not in self.role_names:
            self.role_names.append(role)

        # 确保数据结构完整
        if role not in self.data["角色"]:
            self.data["角色"].append(role)
            for class_name in self.class_names:
                self.data[class_name].append(0)

        # 找到该角色的索引
        role_index = self.data["角色"].index(role)
        self.data[classification][role_index] += 1

    def export_to_excel(self, filename: str):
        """
        导出统计数据到 Excel 文件

        Args:
            filename: 文件名
        """
        try:
            import pandas as pd
            df = pd.DataFrame(self.data)
            df.to_excel(filename, index=False, engine='openpyxl')
            print(f"[OK] 统计数据已导出到: {filename}")
        except ImportError:
            print("[WARNING] 未安装 pandas 或 openpyxl，无法导出 Excel")
            print("请运行: pip install pandas openpyxl")
        except Exception as e:
            print(f"[ERROR] 导出 Excel 失败: {e}")

    def print_statistics(self):
        """打印统计数据"""
        print("\n📊 对话行为统计：")
        print("-" * 80)
        print(f"{'角色':<15} ", end="")
        for class_name in self.class_names:
            print(f"{class_name:<12} ", end="")
        print()
        print("-" * 80)

        for i, role in enumerate(self.data["角色"]):
            print(f"{role:<15} ", end="")
            for class_name in self.class_names:
                count = self.data[class_name][i]
                print(f"{count:<12} ", end="")
            print()
        print("-" * 80)
