"""
改进版：更自然的课堂讨论系统
1. 学生语言更口语化、真实化
2. 识别老师的"讨论引导"信号，触发连续学生互动
"""

import autogen
from autogen import config_list_from_json, GroupChat, GroupChatManager
import random
import re
import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv() # 读取 .env 文件中的环境变量

config_list = [
    {
        "model": os.getenv("OPENAI_MODEL_NAME"),
        "api_key": os.getenv("OPENAI_API_KEY"),
        "base_url": os.getenv("OPENAI_BASE_URL"),
    }
]

# --- 增强的 GroupChat ---
class EnhancedClassroomChat(GroupChat):
    """增强的课堂讨论，支持学生间互动"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.interaction_matrix = {}
        self.silence_count = {}
        self.discussion_mode = False  # 新增：讨论模式标记
        self.discussion_rounds = 0    # 新增：讨论轮数计数
    
    def select_speaker(self, last_speaker: autogen.Agent, selector: autogen.Agent):
        """智能选择下一个发言人"""
        messages = self.messages
        agents = self.agents
        
        if len(messages) <= 1:
            return self._get_agent_by_name("Teacher")
        
        last_message = messages[-1]
        last_content = last_message.get("content", "").strip()
        
        # 异常检测
        if len(last_content) < 5 or last_content == last_speaker.name or last_content.startswith(":"):
            print(f"[警告] 检测到异常消息: '{last_content[:50]}'")
            eligible = [a for a in agents if a != last_speaker and a.name != "Coordinator"]
            selected = random.choice(eligible) if eligible else None
            if selected:
                print(f"[调度日志] 跳过异常，选择: {selected.name}")
            return selected
        
        # 结束检测
        if any(kw in last_content for kw in ["下课", "讨论结束"]):
            print(f"[调度日志] 检测到结束标记")
            return None
        
        # 更新沉默计数
        for agent in agents:
            if agent.name.startswith("Student"):
                if agent == last_speaker:
                    self.silence_count[agent.name] = 0
                else:
                    self.silence_count[agent.name] = self.silence_count.get(agent.name, 0) + 1
        
        # === 核心改进：识别"讨论模式"触发信号 ===
        discussion_triggers = [
            "互相讨论", "相互讨论", "大家讨论", "一起讨论", 
            "互相交流", "相互交流", "大家交流", "一起交流",
            "你们觉得", "大家觉得", "同学们觉得"
        ]
        
        if last_speaker.name == "Teacher":
            # 检测老师是否在鼓励讨论
            if any(trigger in last_content for trigger in discussion_triggers):
                self.discussion_mode = True
                self.discussion_rounds = 0
                print(f"[调度日志] ✨ 进入讨论模式")
        
        # === 讨论模式下的特殊处理 ===
        if self.discussion_mode:
            self.discussion_rounds += 1
            
            # 在讨论模式下，优先让学生间互动（2-4轮）
            if self.discussion_rounds <= 4:
                students = [a for a in agents if a.name.startswith("Student")]
                
                # 如果上一个是学生，选择另一个学生
                if last_speaker.name.startswith("Student"):
                    other_students = [s for s in students if s != last_speaker]
                    if other_students:
                        # 优先选择沉默时间长或互动少的学生
                        selected = max(other_students, 
                                     key=lambda a: self.silence_count.get(a.name, 0))
                        print(f"[调度日志] 🗣️ 讨论模式-学生接力: {selected.name}")
                        self._record_interaction(last_speaker.name, selected.name)
                        return selected
                
                # 如果上一个是老师，随机选一个学生
                elif students:
                    selected = random.choice(students)
                    print(f"[调度日志] 🗣️ 讨论模式-学生发言: {selected.name}")
                    return selected
            else:
                # 讨论进行3-4轮后，让老师总结
                self.discussion_mode = False
                self.discussion_rounds = 0
                teacher = self._get_agent_by_name("Teacher")
                print(f"[调度日志] ⬅️ 退出讨论模式，老师总结")
                return teacher
        
        # === 常规调度逻辑 ===
        
        # 规则1：检测学生间的互相点名
        student_names = [a.name for a in agents if a.name.startswith("Student")]
        for student_name in student_names:
            patterns = [
                rf"{student_name}[，,：:]\s*(?:你|请|能否)",
                rf"@{student_name}",
                rf"{student_name}同学"
            ]
            for pattern in patterns:
                if re.search(pattern, last_content):
                    agent = self._get_agent_by_name(student_name)
                    if agent and agent != last_speaker:
                        print(f"[调度日志] 👉 学生点名: {student_name}")
                        self._record_interaction(last_speaker.name, student_name)
                        return agent
        
        # 规则2：老师点名或提问
        if last_speaker.name == "Teacher":
            # 检测点名
            for student_name in student_names:
                pattern = rf"{student_name}[，,：:]\s*(?:你|请)"
                if re.search(pattern, last_content):
                    agent = self._get_agent_by_name(student_name)
                    if agent:
                        print(f"[调度日志] 👨‍🏫 老师点名: {student_name}")
                        return agent
            
            # 检测提问
            if any(q in last_content for q in ["?", "？", "吗", "呢", "如何", "为什么"]):
                students = [a for a in agents if a.name.startswith("Student")]
                if students:
                    # 优先选择沉默时间较长的学生
                    students_sorted = sorted(students, 
                                           key=lambda a: self.silence_count.get(a.name, 0), 
                                           reverse=True)
                    selected = students_sorted[0] if random.random() < 0.7 else random.choice(students)
                    silence = self.silence_count.get(selected.name, 0)
                    print(f"[调度日志] ❓ 老师提问 → {selected.name} (沉默{silence}轮)")
                    return selected
        
        # 规则3：学生发言后的处理
        if last_speaker.name.startswith("Student"):
            # 检测是否有讨论触发词
            discussion_keywords = [
                "我觉得", "我认为", "但是", "不过", "如果", "是不是", 
                "会不会", "应该", "可能", "或许", "也许", "你们觉得", "大家觉得"
            ]
            
            has_trigger = any(kw in last_content for kw in discussion_keywords)
            has_question = any(q in last_content for q in ["?", "？", "吗", "呢"])
            
            # 如果学生在抛出话题或疑问
            if has_trigger or has_question:
                # 50%概率让另一个学生回应
                if random.random() < 0.5:
                    other_students = [a for a in agents 
                                    if a.name.startswith("Student") and a != last_speaker]
                    if other_students:
                        selected = min(other_students, 
                                     key=lambda a: self.interaction_matrix.get(
                                         (last_speaker.name, a.name), 0))
                        print(f"[调度日志] 💬 学生互动: {selected.name}")
                        self._record_interaction(last_speaker.name, selected.name)
                        return selected
            
            # 检查是否在回答老师的问题
            teacher_asked = self._check_teacher_question(messages)
            
            if teacher_asked:
                # 60%让老师点评，40%让另一学生补充
                if random.random() < 0.6:
                    teacher = self._get_agent_by_name("Teacher")
                    print(f"[调度日志] 📝 学生回答 → 老师点评")
                    return teacher
                else:
                    other_students = [a for a in agents 
                                    if a.name.startswith("Student") and a != last_speaker]
                    if other_students:
                        selected = random.choice(other_students)
                        print(f"[调度日志] ➕ 学生回答 → 学生补充: {selected.name}")
                        return selected
            else:
                # 学生自由发言，70%让老师引导
                if random.random() < 0.7:
                    teacher = self._get_agent_by_name("Teacher")
                    print(f"[调度日志] 🎓 学生发言 → 老师引导")
                    return teacher
        
        # 规则4：Coordinator 后让老师开始
        if last_speaker.name == "Coordinator":
            teacher = self._get_agent_by_name("Teacher")
            print(f"[调度日志] 🔔 课堂开始 → 老师")
            return teacher
        
        # 默认：优先老师
        if last_speaker.name != "Teacher":
            teacher = self._get_agent_by_name("Teacher")
            if random.random() < 0.6:
                print(f"[调度日志] ⏮️ 默认返回老师")
                return teacher
        
        eligible = [a for a in agents if a != last_speaker and a.name != "Coordinator"]
        selected = random.choice(eligible) if eligible else None
        if selected:
            print(f"[调度日志] 🎲 随机选择: {selected.name}")
        return selected
    
    def _check_teacher_question(self, messages):
        """检查前几轮是否有老师提问"""
        for i in range(max(0, len(messages) - 4), len(messages) - 1):
            if messages[i].get("name") == "Teacher":
                content = messages[i].get("content", "")
                if any(q in content for q in ["?", "？", "吗", "呢"]):
                    return True
        return False
    
    def _record_interaction(self, from_agent: str, to_agent: str):
        """记录学生间互动"""
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
        print("\n📊 学生互动统计：")
        for (from_a, to_a), count in sorted(self.interaction_matrix.items()):
            if from_a.startswith("Student") and to_a.startswith("Student"):
                print(f"  {from_a} → {to_a}: {count}次")


# --- 创建智能体（改进语言风格）---
def create_heterogeneous_agents(config_list):
    """创建异构学生智能体 - 更真实的学生语言"""
    
    # 老师
    teacher = autogen.AssistantAgent(
        name="Teacher",
        system_message="""你是一位物理老师，引导牛顿第一定律的讨论。

教学策略：
1. 提出开放性问题
2. 想要学生互相讨论时，明确说："大家可以互相讨论一下"或"你们互相交流看看"
3. 适时点评和总结
4. 当讨论充分（10轮以上）时说"今天的讨论很精彩，下课！"

发言要求：
- 简洁（2-3句话）
- 点名使用：StudentA、StudentB、StudentC
- 每次发言必须是完整的句子
- 绝对不要在发言开头加":Teacher"或类似前缀

示例发言：
✅ "同学们，牛顿第一定律讲的是什么？StudentA，你来说说看。"
✅ "很好！大家可以互相讨论一下这个问题。"
❌ ":Teacher 这是个很好的思考..."（禁止这种格式）
""",
        llm_config={"config_list": config_list, "temperature": 0.7}
    )
    
    # 学生A：积极但知识不深
    student_a = autogen.AssistantAgent(
        name="StudentA",
        system_message="""你是StudentA，一个活泼积极的高中生。

人格：外向、爱提问、知识中等

语言风格（非常重要）：
- 用口语化的表达："诶，我觉得..."、"这个好像是..."
- 可以有不确定："我不太确定诶"、"是不是这样？"
- 用简单词汇，不要学术化
- 会主动找同学："StudentB你怎么想？"
- 回答简短（1-2句话，20-30字）

示例发言：
✅ "我觉得是物体会保持原来的状态吧？不太确定。"
✅ "诶StudentB，你说的那个例子能再讲讲吗？"
✅ "啊，那如果没有摩擦力会怎样？"
❌ "我认为我们可以探讨一下在不同条件下牛顿定律是否仍然适用"（太学术）

注意：
- 不要说"StudentA："
- 像真实的高中生一样说话
- 知识水平有限，不要说太高深的内容
""",
        llm_config={"config_list": config_list, "temperature": 0.9}
    )
    
    # 学生B：严谨、知识好
    student_b = autogen.AssistantAgent(
        name="StudentB",
        system_message="""你是StudentB，一个认真学习的高中生。

人格：内向、严谨、知识较好

语言风格（非常重要）：
- 回答准确但要口语化："嗯，我觉得应该是..."
- 可以纠正但要委婉："我觉得可能不太对，应该是..."
- 用词准确但不过分学术
- 比较被动，较少主动找人
- 回答稍长但不超过3句话（40-60字）

示例发言：
✅ "嗯，惯性就是物体保持原状态的性质，和质量有关。"
✅ "我觉得这个例子不太准确诶，因为还有摩擦力的作用。"
✅ "这个在相对论里会不一样，不过我们现在学的是经典物理。"
❌ "在极端条件下，例如接近光速的运动，牛顿第一定律就不再完全适用..."（太长太学术）

注意：
- 不要说"StudentB："
- 像一个成绩好的真实高中生
- 可以用物理术语，但要自然融入口语
""",
        llm_config={"config_list": config_list, "temperature": 0.65}
    )
    
    # 学生C：爱质疑、批判性思维
    student_c = autogen.AssistantAgent(
        name="StudentC",
        system_message="""你是StudentC，一个爱提问题的高中生。

人格：外向、喜欢辩论、批判性思维

语言风格（非常重要）：
- 喜欢提反例："可是...这种情况怎么说？"
- 会质疑但不无礼："我有点不同意诶"
- 用口语化表达想法
- 会主动挑战其他同学
- 回答简短有力（1-2句话，20-35字）

示例发言：
✅ "可是如果有摩擦力呢？这样不就不是匀速了吗？"
✅ "我不太同意StudentA说的，我觉得应该考虑..."
✅ "但这个在实际中很难实现吧？真的会这样吗？"
❌ "牛顿第一定律的适用性就变得复杂了，我们需要考虑多种因素..."（太学术）

注意：
- 不要说"StudentC："
- 像一个喜欢刨根问底的真实高中生
- 质疑要有趣，不要太严肃
""",
        llm_config={"config_list": config_list, "temperature": 0.85}
    )
    
    # 协调员
    coordinator = autogen.UserProxyAgent(
        name="Coordinator",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda x: any(kw in x.get("content", "") for kw in ["下课", "讨论结束"]),
        max_consecutive_auto_reply=1,
    )
    
    return teacher, student_a, student_b, student_c, coordinator


# --- 主程序 ---
if __name__ == "__main__":
    # 创建智能体
    teacher, student_a, student_b, student_c, coordinator = create_heterogeneous_agents(config_list)
    
    # 创建群聊
    group_chat = EnhancedClassroomChat(
        agents=[teacher, student_a, student_b, student_c, coordinator],
        messages=[],
        max_round=25,
        speaker_selection_method="manual",
        allow_repeat_speaker=False,
    )
    
    # 创建管理器
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": config_list, "temperature": 0.5}
    )
    
    # 开始讨论
    print("=" * 70)
    print("改进版：自然化课堂讨论实验".center(70))
    print("=" * 70)
    print()
    
    try:
        coordinator.initiate_chat(
            manager,
            message="老师，今天讨论牛顿第一定律。请鼓励学生们互相讨论。"
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
    for agent in [teacher, student_a, student_b, student_c]:
        count = sum(1 for m in group_chat.messages if m.get("name") == agent.name)
        percentage = (count / total_rounds * 100) if total_rounds > 0 else 0
        print(f"  {agent.name:12} {count:2}次 ({percentage:5.1f}%)")
    
    # 学生间互动统计
    group_chat.print_interaction_stats()
    
    # 计算学生间互动占比
    student_interactions = sum(1 for (f, t) in group_chat.interaction_matrix.keys()
                              if f.startswith("Student") and t.startswith("Student"))
    print(f"\n💬 学生间互动次数: {student_interactions}")
    
    if total_rounds > 0:
        interaction_ratio = (student_interactions / total_rounds * 100)
        print(f"📈 学生互动占比: {interaction_ratio:.1f}%")
        
        if interaction_ratio > 30:
            print("✅ 互动质量：优秀")
        elif interaction_ratio > 20:
            print("✅ 互动质量：良好")
        else:
            print("⚠️  互动质量：需改进")