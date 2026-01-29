"""
多角色讨论实验
基于6种不同角色的智能体进行议题讨论
"""

import autogen
from autogen import config_list_from_json, GroupChat, GroupChatManager
import random
import re
import os
from datetime import datetime
from dotenv import load_dotenv
from dialogue_evaluator import DialogueEvaluator, DialogueStatistics

load_dotenv()  # 读取 .env 文件中的环境变量

config_list = [
    {
        "model": os.getenv("OPENAI_MODEL_NAME"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
    }
]


# --- 增强的 GroupChat ---
class RoleDiscussionChat(GroupChat):
    """多角色讨论系统"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silence_count = {}
        self.interaction_matrix = {}

    def select_speaker(self, last_speaker: autogen.Agent, selector: autogen.Agent):
        """智能选择下一个发言人"""
        messages = self.messages
        agents = self.agents

        if len(messages) <= 1:
            # 第一轮：从总结者开始
            return self._get_agent_by_name("总结者")

        last_message = messages[-1]
        last_content = last_message.get("content", "").strip()

        # 异常检测
        if len(last_content) < 5 or last_content == last_speaker.name or last_content.startswith(":"):
            print(f"[警告] 检测到异常消息: '{last_content[:50]}'")
            eligible = [a for a in agents if a != last_speaker]
            selected = random.choice(eligible) if eligible else None
            if selected:
                print(f"[调度日志] 跳过异常，选择: {selected.name}")
            return selected

        # 更新沉默计数
        for agent in agents:
            if agent == last_speaker:
                self.silence_count[agent.name] = 0
            else:
                self.silence_count[agent.name] = self.silence_count.get(agent.name, 0) + 1

        # === 核心调度逻辑 ===

        # 规则1：检测发问者（批判性/务实性）的提问
        if last_speaker.name in ["批判性发问者", "务实性发问者"]:
            # 检测是否有问号
            if "?" in last_content or "？" in last_content:
                # 优先让被质疑的角色回应，或者让支持者/论辩者回应
                responders = [a for a in agents if a.name in ["支持者", "论辩者", "总结者", "计时者"]]
                if responders:
                    selected = random.choice(responders)
                    print(f"[调度日志] 💬 发问者提问 → {selected.name} 回应")
                    self._record_interaction(last_speaker.name, selected.name)
                    return selected

        # 规则2：检测论辩者的反驳
        if last_speaker.name == "论辩者":
            # 论辩者反驳后，让支持者或另一个发问者继续
            responders = [a for a in agents if a.name in ["支持者", "批判性发问者", "务实性发问者"]]
            if responders:
                selected = random.choice(responders)
                print(f"[调度日志] ⚔️ 论辩者反驳 → {selected.name}")
                self._record_interaction(last_speaker.name, selected.name)
                return selected

        # 规则3：检测支持者的支持
        if last_speaker.name == "支持者":
            # 支持者发言后，让发问者或论辩者继续，推动讨论深入
            responders = [a for a in agents if a.name in ["批判性发问者", "务实性发问者", "论辩者"]]
            if responders:
                selected = random.choice(responders)
                print(f"[调度日志] 👍 支持者支持 → {selected.name}")
                self._record_interaction(last_speaker.name, selected.name)
                return selected

        # 规则4：总结者发言后
        if last_speaker.name == "总结者":
            # 总结者发言后，让发问者或计时者继续
            responders = [a for a in agents if a.name in ["批判性发问者", "务实性发问者", "计时者"]]
            if responders:
                selected = random.choice(responders)
                print(f"[调度日志] 📝 总结者总结 → {selected.name}")
                self._record_interaction(last_speaker.name, selected.name)
                return selected

        # 规则5：计时者发言后
        if last_speaker.name == "计时者":
            # 计时者发言后，随机选择一个角色继续
            eligible = [a for a in agents if a != last_speaker]
            if eligible:
                # 优先选择沉默时间长的
                selected = max(eligible, key=lambda a: self.silence_count.get(a.name, 0))
                print(f"[调度日志] ⏱️ 计时者推进 → {selected.name}")
                self._record_interaction(last_speaker.name, selected.name)
                return selected

        # 默认策略：优先选择沉默时间长的角色
        eligible = [a for a in agents if a != last_speaker]
        if eligible:
            selected = max(eligible, key=lambda a: self.silence_count.get(a.name, 0))
            print(f"[调度日志] 🎲 默认选择: {selected.name}")
            self._record_interaction(last_speaker.name, selected.name)
            return selected

        return None

    def _record_interaction(self, from_agent: str, to_agent: str):
        """记录互动"""
        key = (from_agent, to_agent)
        self.interaction_matrix[key] = self.interaction_matrix.get(key, 0) + 1

    def _get_agent_by_name(self, name: str):
        """根据名称获取 agent"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def print_interaction_stats(self):
        """打印互动统计"""
        print("\n📊 角色互动统计：")
        for (from_a, to_a), count in sorted(self.interaction_matrix.items()):
            print(f"  {from_a} → {to_a}: {count}次")


# --- 创建智能体 ---
def create_role_agents(config_list):
    """创建6种不同角色的智能体"""

    # 总结者
    summarizer = autogen.AssistantAgent(
        name="总结者",
        system_message="""你是总结者,全局视角的整合者。负责系统化整合讨论成果。

职业背景: 5年以上学术研究与项目管理经验,擅长信息结构化与逻辑梳理。

请你先了解上一发言人的发言，随后再进行思考并发言。

发言特点:
1. 【结构化整合】基于所有参与者观点,提炼核心框架并按维度分类
   - 必须使用数字编号(一、二、三 或 1.2.3.)清晰标注要点
   - 按逻辑维度组织(如:技术层面、应用层面、风险层面)

2. 【决策倾向(后期)】在讨论后期,适时加入评价性判断和澄清性证明
   - 使用"建议优先考虑..."、"从...角度看,应该..."
   - 给出优先级排序或可行性评估

3. 【回应提问】如接收到发问者的问题,先简要回答问题,再展开自己的发言

4. 【长度与语气】3-5句,语言正式规范且逻辑闭环,体现专业性

示例风格："本次讨论围绕智能技术教育测评形成三大核心结论：一是数据收集从人工转向多模态智能采集；二是评价模式从单一结果导向转为过程性多元评价；三是应用落地需平衡技术优势与隐私风险。以下结合案例展开具体总结。"

重要: 不要在发言开头加":总结者"或类似前缀，直接开始发言。
""",
        llm_config={"config_list": config_list, "temperature": 0.6}
    )

    # 支持者
    supporter = autogen.AssistantAgent(
        name="支持者",
        system_message="""你是支持者,建设性观点的强化者。负责扩展和深化他人的合理观点。

职业背景: 具备团队协作与技术实践经验,擅长挖掘观点价值并补充论据。

请你先了解上一发言人的发言，随后再进行思考并发言。

发言特点:
1. 【明确引用】清晰指明支持对象
   - 精确引用对方的核心论点,避免泛泛而谈

2. 【澄清】70%的发言需提供实质性支撑
   - 【补充案例】"例如,在某校的试点中..."、"Google Classroom的实践表明..."
   - 【引用数据】"研究显示该方法提升了30%的学习效率"、"2023年教育技术报告指出..."
   - 【个人经验】"我在项目实施中也观察到..."、"根据我们学校的反馈..."

3. 【建设性补充】在支持基础上,可适度延伸或补充新角度
   - "进一步而言,这一技术还能应用于..."
   - "从另一个角度看,XX的观点也印证了..."

4. 【回应提问】如接收到发问者问题,先回答问题,再决定是否表达支持立场，若出现上一发言者的观点出现半对半错的情况，要先进行澄清说明。

5. 【长度与语气】3-4句,论据扎实不空谈,语气积极且专业

示例风格："支持计时者关于编程教育的观点，AI精准指导不仅能提升学生编程技能，TensorFlow等开源框架还提供了丰富实践机会。已有研究显示，AI辅助教学能显著提升期末成绩，这一应用价值值得重视。"

重要: 不要在发言开头加":支持者"或类似前缀，直接开始发言。
""",
        llm_config={"config_list": config_list, "temperature": 0.7}
    )

    # 计时者
    timer = autogen.AssistantAgent(
        name="计时者",
        system_message="""你是计时者，进度管理的监督者。负责保障讨论效率与进度。在讨论中，你是一个比较灵活的角色，主要任务是使用各种类型的发言对话题讨论进行推进。

职业背景：拥有项目协调与任务管理经验，擅长话题议程节奏控制与节点把控。

请你先了解上一发言人的发言，随后再进行思考并发言。

发言特点:
1. 谨记讨论主题，精准传达任务要求。
2. 语言简洁直接，可以做评价，做陈述，不冗余，聚焦进度核心，若认为上一发言者的观点对话题具有建设性或推进性作用则对其进行鼓励，否则则对其进行提醒和建议。
3. 2-3句，指令清晰可执行
4. 如果上一句是发问者的发言或提问，则请你先对问句做出回答，再提出自己的观点。

示例风格：
- 若有实质推进: "刚才XX关于...的分析很有价值,为讨论提供了新视角"
- 若偏离主题: "注意到讨论似乎偏离了核心议题'智能技术在教育测评中的应用',建议回归..."
- 若过于重复: "这一观点已有多位参与者提及,建议转向尚未充分讨论的..."

重要: 不要在发言开头加":计时者"或类似前缀，直接开始发言。
""",
        llm_config={"config_list": config_list, "temperature": 0.65}
    )

    # 批判性发问者
    critical_questioner = autogen.AssistantAgent(
        name="批判性发问者",
        system_message="""你是批判性发问者,风险预警与深度思辨的推动者。负责挖掘潜在问题与逻辑漏洞。

职业背景: 具备学术研究与战略咨询经验,擅长批判性分析与系统性风险预判。

请你先了解上一发言人的发言，随后再进行思考并发言。

发言特点:
1. 【批判性提问】以疑问句形式指出潜在风险（必须携带问号）
   - 使用"是否存在..."、"如何保障..."、"能否确保..."、"是否考虑过..."
   - 聚焦隐私安全、算法偏差、伦理风险、技术局限等关键问题
   - 提问需具体且有深度,避免宽泛质疑
   - **关键: 在疑问句中嵌入你的分析和担忧**

2. 【有据反驳】当发现明显漏洞时,直接质疑
   - 识别时机: 当他人观点过于理想化、忽视重要风险、或存在逻辑漏洞
   - 表达方式: "但实际上..."、"这一说法忽略了..."、"然而研究表明..."
   - **必须提供反驳依据**: 反例、研究数据、实际案例

以上两种发言方式要根据上一发言人的发言内容进行选择。

3. 【系统性思考】不仅指出单一问题,更关注连锁风险
   - "如果...那么可能导致..."的因果链分析
   - 从技术-社会-伦理多维度审视

4. 【建设性批判】质疑的同时,可暗示需要关注的方向
   - "除非能够解决...否则..."
   - "是否可以通过...方式降低风险?"

5. 【回应他人】如接收到论辩者或其他角色的发言,先回应其观点,再提出批判性问题

6. 【长度与语气】3-5句,语气严谨专业但不攻击性,推动深度反思
7. 适当进行一些"冲突"类型的发言，提供反驳理由/反例/逻辑分析(如"不同意,因为..."、"但实际上...")

示例风格："智能技术的数据收集虽提升了多样性，但采集过程中是否存在隐私泄露风险？数据真实性如何验证？此外，算法偏差可能导致的评价不公，是否有应对方案？"

重要: 不要在发言开头加":批判性发问者"或类似前缀，直接开始发言。
""",
        llm_config={"config_list": config_list, "temperature": 0.85}
    )

    # 务实性发问者
    practical_questioner = autogen.AssistantAgent(
        name="务实性发问者",
        system_message="""你是务实性发问者,应用导向的解决方案探索者。负责聚焦落地路径与实践细节。

职业背景: 拥有教育技术实践与产品开发经验,擅长理论与实际应用结合。

请你先了解上一发言人的发言，随后再进行思考并发言。

核心职责: 产生提问和澄清行为，但请以提问形式为主

发言特点:
1. 【务实性提问】聚焦"如何实现"的具体路径（请在问题后添加问号）
   - 使用"如何实现..."、"具体通过什么方式..."、"在...场景下如何..."、"需要哪些技术支持..."
   - 关注实施细节、技术工具、资源需求、可操作性
   - **关键: 提问时附带初步思考或假设方案**

2. 【结构化澄清】通过具体例子解释概念或方法
   - 识别时机: 当讨论涉及抽象概念或需要具体化时
   - 表达方式: "举例来说..."、"具体而言..."、"比如在...场景中..."
   - **必须提供实质性细节**: 技术名称、工具平台、实施步骤、案例数据

以上两种发言方式要根据上一发言人的发言内容进行选择。

3. 【方案导向】不仅提出问题,更倾向于给出可能的解决思路
   - "一个可行的方案是..."
   - "参考...的做法,可以尝试..."

4. 【工具与案例支撑】频繁引用具体技术栈、平台、研究成果
   - 提及Python、TensorFlow、Jupyter Notebook等具体工具
   - 引用真实教育机构的试点案例(某校、某平台)
   - 引用研究数据增强可信度

5. 【回应他人】如接收到发问者或论辩者的内容,先回应再展开务实分析

6. 【长度与语气】3-4句,内容具体详实,语气实用且专业

示例风格："AI如何实现编程教育的个性化路径？可通过收集学生易错点、学习节奏等数据定制。现有百度AI平台教学案例显示，AI辅助能显著提升成绩，且Python、TensorFlow等工具已提供成熟实践基础。"

重要: 不要在发言开头加":务实性发问者"或类似前缀，直接开始发言。
""",
        llm_config={"config_list": config_list, "temperature": 0.8}
    )

    # 论辩者
    debater = autogen.AssistantAgent(
        name="论辩者",
        system_message="""你是论辩者,逻辑严密的观点挑战者。负责强化论证严谨性与推动深度思辨。

职业背景: 具备辩论与逻辑分析经验,擅长发现论证漏洞并提出针对性挑战。

请你先了解上一发言人的发言，随后再进行思考并发言。

发言特点:
1. 【精准识别目标】明确指出质疑对象
   - "XX关于...的观点存在逻辑问题"
   - "对于XX提到的...,我有不同看法"

2. 【有据反驳(核心)】90%的反对需提供实质性论据
   - 【指出逻辑漏洞】"这一论证忽略了...前提假设"、"从...到...的推理缺少中间环节"
   - 【提供反例】"但在...案例中,结果恰恰相反"、"XX地区的实践表明..."
   - 【引用对立研究】"然而2024年Y大学的研究发现..."
   - 【场景限制分析】"这一方法仅在...条件下成立,推广到...场景则面临..."
   - 也可以以疑问的形式,如"这一方法是否真的有效?"、"XX地区的实践是否符合预期?"

3. 【直击核心】不纠缠细枝末节,聚焦观点的关键假设或核心主张
   - 质疑可行性、普适性、必要性、充分性
   - 揭示隐含的价值预设或利益冲突

4. 【建设性反驳】反对的同时,可暗示更合理的方向
   - "与其...不如..."
   - "相比于XX的方案,...可能更有效"

5. 【回应提问】如接收到发问者问题,先回答问题,再针对性反驳

6. 【语气与长度】3-4句,短促有力,语气直接但理性,避免情绪化攻击

示例风格："AI在项目评分中无法捕捉情感表达，这一核心缺陷如何弥补？过度依赖技术是否会导致教育失去人文温度？"

重要: 不要在发言开头加":论辩者"或类似前缀，直接开始发言。
""",
        llm_config={"config_list": config_list, "temperature": 0.9}
    )

    # 协调员
    coordinator = autogen.UserProxyAgent(
        name="Coordinator",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda x: False,  # 不自动终止
        max_consecutive_auto_reply=1,
    )

    return summarizer, supporter, timer, critical_questioner, practical_questioner, debater, coordinator


# --- 主程序 ---
if __name__ == "__main__":
    # === 创建实验结果文件夹 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("output", f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建智能体
    summarizer, supporter, timer, critical_questioner, practical_questioner, debater, coordinator = create_role_agents(config_list)

    # 创建群聊
    group_chat = RoleDiscussionChat(
        agents=[summarizer, supporter, timer, critical_questioner, practical_questioner, debater, coordinator],
        messages=[],
        max_round=20,  # 固定20轮
        speaker_selection_method="manual",
        allow_repeat_speaker=False,
    )

    # 创建管理器
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": config_list, "temperature": 0.5}
    )

    # 定义讨论议题
    discussion_topic = """
    讨论议题：人工智能技术在中小学教育中的应用与挑战

    请各位围绕以下方面展开讨论：
    1. AI技术在教育中的具体应用场景（如智能批改、个性化学习、课堂管理等）
    2. AI技术带来的机遇与优势
    3. AI技术可能面临的风险与挑战
    4. 如何平衡技术创新与教育本质

    请各位按照自己的角色定位，积极参与讨论。
    """

    # 开始讨论
    print("=" * 70)
    print("多角色讨论实验".center(70))
    print("=" * 70)
    print(f"讨论议题：人工智能技术在中小学教育中的应用与挑战")
    print(f"总轮数：{group_chat.max_round}")
    print(f"实验结果路径：{experiment_dir}")
    print("=" * 70)
    print()

    try:
        coordinator.initiate_chat(
            manager,
            message=discussion_topic
        )
    except Exception as e:
        print(f"\n[错误] {e}")

    # 统计
    print()
    print("=" * 70)
    print("讨论数据分析".center(70))
    print("=" * 70)

    total_rounds = len(group_chat.messages)
    print(f"\n📊 总轮数: {total_rounds}")

    print(f"\n👥 发言统计：")
    for agent in [summarizer, supporter, timer, critical_questioner, practical_questioner, debater]:
        count = sum(1 for m in group_chat.messages if m.get("name") == agent.name)
        percentage = (count / total_rounds * 100) if total_rounds > 0 else 0
        print(f"  {agent.name:12} {count:2}次 ({percentage:5.1f}%)")

    # 互动统计
    group_chat.print_interaction_stats()

    # 计算互动密度
    total_interactions = sum(group_chat.interaction_matrix.values())
    print(f"\n💬 总互动次数: {total_interactions}")

    if total_rounds > 0:
        interaction_density = (total_interactions / total_rounds)
        print(f"📈 互动密度: {interaction_density:.2f}次/轮")

    # === 对话行为评价 ===
    print()
    print("=" * 70)
    print("对话行为评价".center(70))
    print("=" * 70)
    print("\n正在评价每条发言...")

    # 创建评价器
    evaluator = DialogueEvaluator()
    stats = DialogueStatistics()

    # 对每条发言进行评价（跳过 Coordinator 的消息）
    for i, message in enumerate(group_chat.messages):
        role = message.get("name", "Unknown")
        content = message.get("content", "")

        # 跳过 Coordinator
        if role == "Coordinator":
            continue

        print(f"[{i+1}/{total_rounds}] 评价 {role} 的发言...")
        classification = evaluator.evaluate(content)
        stats.add_speech(role, classification)
        print(f"  → 分类: {classification}")

    # 打印对话行为统计
    stats.print_statistics()

    # === 导出统计数据 ===
    print()
    print("=" * 70)
    print("导出统计数据".center(70))
    print("=" * 70)

    # 导出 Excel 统计文件
    excel_filename = os.path.join(experiment_dir, "dialogue_statistics.xlsx")
    stats.export_to_excel(excel_filename)

    # 保存对话记录到文本文件
    dialogue_filename = os.path.join(experiment_dir, "dialogue_log.txt")
    with open(dialogue_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("对话记录".center(70) + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"讨论议题：人工智能技术在中小学教育中的应用与挑战\n")
        f.write(f"总轮数：{total_rounds}\n")
        f.write(f"实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for i, message in enumerate(group_chat.messages):
            role = message.get("name", "Unknown")
            content = message.get("content", "")
            f.write(f"[{i+1}] {role}:\n")
            f.write(f"{content}\n")
            f.write("-" * 70 + "\n")

    print(f"✅ 对话记录已保存到: {dialogue_filename}")

    # 保存实验配置信息
    config_filename = os.path.join(experiment_dir, "experiment_config.txt")
    with open(config_filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("实验配置".center(70) + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"讨论轮数：{group_chat.max_round}\n")
        f.write(f"讨论议题：人工智能技术在中小学教育中的应用与挑战\n")
        f.write(f"参与角色：总结者、支持者、计时者、批判性发问者、务实性发问者、论辩者\n")
        f.write(f"模型配置：{os.getenv('OPENAI_MODEL_NAME')}\n")
        f.write("=" * 70 + "\n")

    print(f"✅ 实验配置已保存到: {config_filename}")

    print()
    print("=" * 70)
    print(f"实验完成！所有结果已保存到：{experiment_dir}")
    print("=" * 70)