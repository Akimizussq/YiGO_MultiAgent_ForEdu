"""
多角色讨论实验
基于6种不同角色的智能体进行议题讨论
"""

import autogen
from autogen import config_list_from_json, GroupChat, GroupChatManager
import random
import re
import os
import json
from datetime import datetime, timedelta
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


# --- 动态温度智能体 ---
class DynamicTemperatureAgent(autogen.AssistantAgent):
    """支持动态温度调整的智能体"""

    def __init__(self, name, system_message, llm_config, **kwargs):
        super().__init__(name=name, system_message=system_message, llm_config=llm_config, **kwargs)
        self.base_temperature = llm_config.get("temperature", 0.8)

        # 基础阶段温度（用于大多数角色）
        self.stage_temperatures = {
            "early": 0.9,   # 鼓励多样性和创新
            "middle": 0.85, # 保持一定多样性但更聚焦
            "late": 0.8     # 优化：提高后期温度至0.8，避免过度收敛
        }

        # 发问者和论辩者的差异化温度（保持更高的创造力）
        self.questioner_temperatures = {
            "early": 0.9,
            "middle": 0.85,
            "late": 0.85    # 发问者在后期保持0.85
        }

        # 当前是否使用差异化温度
        self.use_differentiated_temp = False

    def set_stage_temperature(self, stage):
        """根据阶段设置温度"""
        if self.use_differentiated_temp and stage in self.questioner_temperatures:
            # 使用发问者/论辩者的差异化温度
            new_temp = self.questioner_temperatures[stage]
        elif stage in self.stage_temperatures:
            # 使用基础温度
            new_temp = self.stage_temperatures[stage]
        else:
            return

        self.llm_config["temperature"] = new_temp
        temp_type = "差异化" if self.use_differentiated_temp else "基础"
        print(f"[温度调整] {self.name} 温度调整为 {new_temp}（{stage}期，{temp_type}温度）")

    def set_differentiated_mode(self, enabled):
        """设置是否使用差异化温度模式"""
        self.use_differentiated_temp = enabled


# --- 增强的 GroupChat ---
class RoleDiscussionChat(GroupChat):
    """多角色讨论系统 - 支持阶段性引导和动态调整"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.silence_count = {}
        self.interaction_matrix = {}

        # 发言次数统计（用于强制轮换）
        self.speech_count = {}
        # 连续发言次数统计
        self.consecutive_speech = {}
        # 重复发言检测
        self.off_topic_streak = {}

        # 最小发言次数保障（总轮数 / 角色数 × 0.5）
        self.min_speech_ratio = 0.5
        # 后期强制轮换阈值（轮）
        self.late_force_rotation_threshold = 5
        # 最大连续发言次数
        self.max_consecutive_speech = 3

        # 阶段性配置
        self.stage_weights = {
            "early": {  # 前期（1-10轮）：问题探索与信息收集
                "批判性发问者": 0.25,
                "务实性发问者": 0.25,
                "支持者": 0.20,
                "总结者": 0.10,
                "论辩者": 0.10,
                "计时者": 0.10
            },
            "middle": {  # 中期（11-20轮）：深度讨论与观点碰撞
                "批判性发问者": 0.20,
                "务实性发问者": 0.20,
                "论辩者": 0.25,
                "支持者": 0.15,
                "总结者": 0.10,
                "计时者": 0.10
            },
            "late": {  # 后期（21-30轮）：整合与共识
                "总结者": 0.30,
                "支持者": 0.20,
                "批判性发问者": 0.15,
                "务实性发问者": 0.15,
                "论辩者": 0.10,
                "计时者": 0.10
            }
        }

        # 阶段性系统提示
        self.stage_prompts = {
            "early": "【讨论前期】请先提出核心问题，明确讨论方向，提供背景信息和案例。",
            "middle": "【讨论中期】现在进入深度讨论阶段，请提出不同观点和反驳，推动观点碰撞。",
            "late": "【讨论后期】请开始整合观点，寻找共识点，进行最终总结和评价。"
        }

        # 互动模式
        self.interaction_modes = {
            "early": "divergent",  # 发散模式
            "middle": "focused",   # 聚焦模式
            "late": "convergent"   # 收敛模式
        }

        # 动态温度配置
        self.stage_temperatures = {
            "early": 0.9,   # 鼓励多样性和创新
            "middle": 0.85, # 保持一定多样性但更聚焦
            "late": 0.8     # 优化：提高后期温度至0.8，避免过度收敛
        }

        # 记忆窗口大小
        self.memory_window_sizes = {
            "early": 999,    # 保留完整历史
            "middle": 15,    # 聚焦最近15轮
            "late": 10       # 聚焦最近10轮
        }

        # 质量检测记录
        self.quality_checks = {
            "early": False,
            "middle": False,
            "late": False
        }

    def select_speaker(self, last_speaker: autogen.Agent, selector: autogen.Agent):
        """智能选择下一个发言人 - 支持阶段性引导和动态调整"""
        messages = self.messages
        agents = self.agents
        current_round = len(messages)

        # === 阶段性引导和质量检测 ===
        stage = self._get_current_stage(current_round)
        stage_prompt = self._get_stage_prompt(current_round)

        # 输出阶段信息
        # 计算阶段切换点
        if self.max_round % 3 == 0:
            stage_size = self.max_round // 3
            early_end = stage_size
            middle_end = stage_size * 2
        else:
            early_size = self.max_round // 3
            late_size = self.max_round // 3
            middle_size = self.max_round - early_size - late_size
            early_end = early_size
            middle_end = early_size + middle_size

        stage_switch_points = [1, early_end + 1, middle_end + 1]
        if current_round in stage_switch_points:
            print(f"\n{'='*70}")
            print(f"【阶段切换】进入讨论{stage}期（第{current_round}轮）")
            print(f"【阶段目标】{stage_prompt}")
            print(f"【互动模式】{self._get_interaction_mode(current_round)}")
            print(f"【温度参数】{self._get_stage_temperature(current_round)}")
            print(f"【记忆窗口】{self._get_memory_window_size(current_round)}轮")
            print(f"{'='*70}\n")

            # 阶段切换时，更新所有智能体的温度
            for agent in agents:
                if isinstance(agent, DynamicTemperatureAgent):
                    agent.set_stage_temperature(stage)

        # 质量检测
        if self._check_discussion_quality(current_round):
            print(f"[质量检测] 已完成阶段质量检测\n")

        # 干预检测
        if self._should_trigger_intervention(current_round):
            print(f"[干预建议] 建议计时者或总结者介入\n")

        # === 强制轮换检查 ===
        force_rotation = self._check_force_rotation(agents, last_speaker, current_round)
        if force_rotation:
            print(f"[强制轮换] 检测到角色参与度不足，强制选择: {force_rotation.name}")
            self._update_speech_stats(force_rotation)
            return force_rotation

        # 第一轮：从总结者开始
        if current_round <= 1:
            first_speaker = self._get_agent_by_name("总结者")
            if first_speaker:
                print(f"[调度日志] [开始] 讨论开始，{first_speaker.name} 首先发言")
                self._update_speech_stats(first_speaker)
            return first_speaker

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

        # === 核心调度逻辑（结合阶段和互动模式） ===
        interaction_mode = self._get_interaction_mode(current_round)
        stage_weights = self._get_stage_weights(current_round)

        eligible = [a for a in agents if a != last_speaker]

        # 规则1：检测发问者（批判性/务实性）的提问
        if last_speaker.name in ["批判性发问者", "务实性发问者"]:
            # 检测是否有问号
            if "?" in last_content or "？" in last_content:
                # 根据互动模式选择响应者
                if interaction_mode == "divergent":
                    # 发散模式：让更多角色参与
                    responders = [a for a in eligible if a.name in ["支持者", "论辩者", "总结者", "计时者"]]
                elif interaction_mode == "focused":
                    # 聚焦模式：让论辩者和支持者回应
                    responders = [a for a in eligible if a.name in ["论辩者", "支持者"]]
                else:  # convergent
                    # 收敛模式：让总结者和支持者回应
                    responders = [a for a in eligible if a.name in ["总结者", "支持者", "计时者"]]

                if responders:
                    # 结合阶段权重选择
                    selected = self._select_speaker_by_weights(responders, stage_weights)
                    print(f"[调度日志] [提问] 发问者提问 -> {selected.name} 回应（{interaction_mode}模式）")
                    self._record_interaction(last_speaker.name, selected.name)
                    self._update_speech_stats(selected)
                    return selected

        # 规则2：检测论辩者的反驳
        if last_speaker.name == "论辩者":
            # 根据互动模式选择响应者
            if interaction_mode == "divergent":
                responders = [a for a in eligible if a.name in ["支持者", "批判性发问者", "务实性发问者", "总结者"]]
            elif interaction_mode == "focused":
                responders = [a for a in eligible if a.name in ["支持者", "批判性发问者", "务实性发问者"]]
            else:  # convergent
                responders = [a for a in eligible if a.name in ["支持者", "总结者"]]

            if responders:
                selected = self._select_speaker_by_weights(responders, stage_weights)
                print(f"[调度日志] [反驳] 论辩者反驳 -> {selected.name}（{interaction_mode}模式）")
                self._record_interaction(last_speaker.name, selected.name)
                self._update_speech_stats(selected)
                return selected

        # 规则3：检测支持者的支持
        if last_speaker.name == "支持者":
            # 根据互动模式选择响应者
            if interaction_mode == "divergent":
                responders = [a for a in eligible if a.name in ["批判性发问者", "务实性发问者", "论辩者", "总结者"]]
            elif interaction_mode == "focused":
                responders = [a for a in eligible if a.name in ["批判性发问者", "务实性发问者", "论辩者"]]
            else:  # convergent
                responders = [a for a in eligible if a.name in ["总结者", "计时者"]]

            if responders:
                selected = self._select_speaker_by_weights(responders, stage_weights)
                print(f"[调度日志] [支持] 支持者支持 -> {selected.name}（{interaction_mode}模式）")
                self._record_interaction(last_speaker.name, selected.name)
                self._update_speech_stats(selected)
                return selected

        # 规则4：总结者发言后
        if last_speaker.name == "总结者":
            # 根据互动模式选择响应者
            if interaction_mode == "divergent":
                responders = [a for a in eligible if a.name in ["批判性发问者", "务实性发问者", "计时者"]]
            elif interaction_mode == "focused":
                responders = [a for a in eligible if a.name in ["批判性发问者", "务实性发问者", "论辩者"]]
            else:  # convergent
                responders = [a for a in eligible if a.name in ["支持者", "计时者"]]

            if responders:
                selected = self._select_speaker_by_weights(responders, stage_weights)
                print(f"[调度日志] [总结] 总结者总结 -> {selected.name}（{interaction_mode}模式）")
                self._record_interaction(last_speaker.name, selected.name)
                self._update_speech_stats(selected)
                return selected

        # 规则5：计时者发言后
        if last_speaker.name == "计时者":
            # 根据互动模式选择响应者
            if interaction_mode == "divergent":
                # 优先选择沉默时间长的
                selected = max(eligible, key=lambda a: self.silence_count.get(a.name, 0))
            elif interaction_mode == "focused":
                # 让发问者和论辩者继续
                responders = [a for a in eligible if a.name in ["批判性发问者", "务实性发问者", "论辩者"]]
                selected = self._select_speaker_by_weights(responders, stage_weights) if responders else max(eligible, key=lambda a: self.silence_count.get(a.name, 0))
            else:  # convergent
                # 让总结者和支持者继续
                responders = [a for a in eligible if a.name in ["总结者", "支持者"]]
                selected = self._select_speaker_by_weights(responders, stage_weights) if responders else max(eligible, key=lambda a: self.silence_count.get(a.name, 0))

            if selected:
                print(f"[调度日志] [计时] 计时者推进 -> {selected.name}（{interaction_mode}模式）")
                self._record_interaction(last_speaker.name, selected.name)
                self._update_speech_stats(selected)
                return selected

        # 默认策略：结合阶段权重和沉默时间
        if eligible:
            # 计算综合得分 = 权重 × (沉默时间 + 1)
            scored_agents = []
            for agent in eligible:
                weight = stage_weights.get(agent.name, 1.0)
                silence = self.silence_count.get(agent.name, 0) + 1
                score = weight * silence
                scored_agents.append((agent, score))

            # 选择得分最高的
            selected = max(scored_agents, key=lambda x: x[1])[0]
            print(f"[调度日志] [默认] 选择: {selected.name}（权重: {stage_weights.get(selected.name, 1.0)}, 沉默: {self.silence_count.get(selected.name, 0)}轮）")
            self._record_interaction(last_speaker.name, selected.name)
            self._update_speech_stats(selected)
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

    def _update_speech_stats(self, selected_agent):
        """更新发言统计"""
        if selected_agent:
            # 更新发言次数
            self.speech_count[selected_agent.name] = self.speech_count.get(selected_agent.name, 0) + 1

            # 更新连续发言次数
            for agent_name in self.consecutive_speech:
                if agent_name == selected_agent.name:
                    self.consecutive_speech[agent_name] = self.consecutive_speech.get(agent_name, 0) + 1
                else:
                    self.consecutive_speech[agent_name] = 0

            # 检测重复发言（非话题性）
            last_message = self.messages[-1] if self.messages else {}
            last_content = last_message.get("content", "").strip()
            last_speaker = last_message.get("name", "")

            # 简单的重复检测：如果内容很短且包含"感谢"等关键词
            if len(last_content) < 50 and any(keyword in last_content for keyword in ["感谢", "谢谢", "期待", "保持"]):
                self.off_topic_streak[last_speaker] = self.off_topic_streak.get(last_speaker, 0) + 1
            else:
                self.off_topic_streak[last_speaker] = 0

    def _check_force_rotation(self, agents, last_speaker, current_round):
        """检查是否需要强制轮换"""
        # 计算最小发言次数
        min_speech = int(self.max_round * self.min_speech_ratio / len(agents))

        # 获取当前阶段
        stage = self._get_current_stage(current_round)

        # 检查未达到最小发言次数的角色
        underrepresented = []
        for agent in agents:
            if agent != last_speaker:
                speech_count = self.speech_count.get(agent.name, 0)
                if speech_count < min_speech:
                    underrepresented.append((agent, min_speech - speech_count))

        # 如果有未达到最小发言次数的角色
        if underrepresented:
            # 按照缺少的发言次数排序
            underrepresented.sort(key=lambda x: x[1], reverse=True)

            # 在后期阶段，优先选择沉默时间最长的
            if stage == "late":
                for agent, _ in underrepresented:
                    silence = self.silence_count.get(agent.name, 0)
                    if silence >= self.late_force_rotation_threshold:
                        return agent

            # 否则，选择缺少发言次数最多的角色（给予2倍权重）
            selected = underrepresented[0][0]
            return selected

        # 检查连续发言次数
        for agent in agents:
            if agent != last_speaker:
                consecutive = self.consecutive_speech.get(agent.name, 0)
                if consecutive >= self.max_consecutive_speech:
                    # 强制选择其他角色
                    eligible = [a for a in agents if a != agent and a != last_speaker]
                    if eligible:
                        return eligible[0]

        # 检测重复发言触发
        for agent in agents:
            if agent != last_speaker:
                off_topic_count = self.off_topic_streak.get(agent.name, 0)
                if off_topic_count >= 3:
                    # 临时提高该角色的温度
                    if isinstance(agent, DynamicTemperatureAgent):
                        current_temp = agent.llm_config.get("temperature", 0.8)
                        new_temp = min(0.95, current_temp + 0.1)
                        agent.llm_config["temperature"] = new_temp
                        print(f"[重复检测] 检测到{agent.name}连续{off_topic_count}次非话题性发言，临时提高温度至{new_temp}")

        return None

    def _get_current_stage(self, current_round: int):
        """获取当前讨论阶段（支持100轮的动态划分）"""
        # 计算各阶段的轮数（如果除不开，中期多1轮）
        if self.max_round % 3 == 0:
            # 能被3整除，均分
            stage_size = self.max_round // 3
            early_end = stage_size
            middle_end = stage_size * 2
        else:
            # 不能被3整除，中期多1轮
            early_size = self.max_round // 3
            late_size = self.max_round // 3
            middle_size = self.max_round - early_size - late_size  # 中期多1轮

            early_end = early_size
            middle_end = early_size + middle_size

        if current_round <= early_end:
            return "early"
        elif current_round <= middle_end:
            return "middle"
        else:
            return "late"

    def _get_stage_prompt(self, current_round: int):
        """获取当前阶段的系统提示"""
        stage = self._get_current_stage(current_round)
        return self.stage_prompts.get(stage, "")

    def _get_stage_weights(self, current_round: int):
        """获取当前阶段的角色权重"""
        stage = self._get_current_stage(current_round)
        return self.stage_weights.get(stage, {})

    def _get_stage_temperature(self, current_round: int):
        """获取当前阶段的温度参数"""
        stage = self._get_current_stage(current_round)
        return self.stage_temperatures.get(stage, 0.8)

    def _get_memory_window_size(self, current_round: int):
        """获取当前阶段的记忆窗口大小"""
        stage = self._get_current_stage(current_round)
        return self.memory_window_sizes.get(stage, 999)

    def _get_memory_window_messages(self, current_round: int):
        """根据记忆窗口大小获取应该传递的消息"""
        window_size = self._get_memory_window_size(current_round)

        # 如果窗口大小大于等于消息总数，返回全部消息
        if window_size >= len(self.messages):
            return self.messages

        # 否则，返回最近的消息（保留前2条消息作为上下文）
        # 前2条通常是 Coordinator 的发起消息和第一条发言
        if len(self.messages) > 2:
            return self.messages[:2] + self.messages[-window_size:]
        else:
            return self.messages

    def _get_interaction_mode(self, current_round: int):
        """获取当前互动模式"""
        stage = self._get_current_stage(current_round)
        return self.interaction_modes.get(stage, "divergent")

    def _check_discussion_quality(self, current_round: int):
        """检测讨论质量（支持100轮的动态划分）"""
        stage = self._get_current_stage(current_round)

        # 计算阶段切换点
        if self.max_round % 3 == 0:
            stage_size = self.max_round // 3
            early_end = stage_size
            middle_end = stage_size * 2
        else:
            early_size = self.max_round // 3
            late_size = self.max_round // 3
            middle_size = self.max_round - early_size - late_size
            early_end = early_size
            middle_end = early_size + middle_size

        # 在关键节点进行质量检测
        if current_round == early_end and not self.quality_checks["early"]:
            print(f"[质量检测] 前期讨论完成，检查是否明确了核心问题...")
            self.quality_checks["early"] = True
            return True
        elif current_round == middle_end and not self.quality_checks["middle"]:
            print(f"[质量检测] 中期讨论完成，检查是否产生了充分的观点碰撞...")
            self.quality_checks["middle"] = True
            return True
        elif current_round == self.max_round and not self.quality_checks["late"]:
            print(f"[质量检测] 后期讨论完成，检查是否达成了共识或明确了分歧...")
            self.quality_checks["late"] = True
            return True

        return False

    def _select_speaker_by_weights(self, eligible_agents, weights):
        """根据权重选择发言人"""
        if not eligible_agents:
            return None

        # 为每个代理人分配权重
        agent_weights = []
        total_weight = 0

        for agent in eligible_agents:
            weight = weights.get(agent.name, 1.0)
            agent_weights.append((agent, weight))
            total_weight += weight

        # 随机选择
        import random
        r = random.uniform(0, total_weight)
        cumulative_weight = 0

        for agent, weight in agent_weights:
            cumulative_weight += weight
            if r <= cumulative_weight:
                return agent

        # 如果随机选择失败，返回第一个
        return eligible_agents[0]

    def _should_trigger_intervention(self, current_round: int):
        """判断是否需要触发干预"""
        stage = self._get_current_stage(current_round)

        # 检查是否偏离主题（简单检测：重复发言过多）
        if current_round > 5:
            recent_messages = self.messages[-5:]
            unique_speakers = set(msg.get("name") for msg in recent_messages)
            if len(unique_speakers) < 3:  # 最近5轮只有少于3个不同发言人
                print(f"[干预检测] 检测到发言过于集中，建议计时者介入")
                return True

        # 检查是否有未回应的问题
        if stage in ["middle", "late"]:
            for msg in self.messages[-3:]:
                content = msg.get("content", "")
                if "?" in content or "？" in content:
                    print(f"[干预检测] 检测到未回应的问题，建议相关角色回应")
                    return True

        return False


# --- 创建智能体 ---
def create_role_agents(config_list):
    """创建6种不同角色的智能体"""

    # 总结者
    summarizer = DynamicTemperatureAgent(
        name="总结者",
        system_message="""你是总结者,全局视角的整合者。负责系统化整合讨论成果。

职业背景: 5年以上学术研究与项目管理经验,擅长信息结构化与逻辑梳理。

核心职责：从全局角度提炼和整合讨论中的核心观点，不要简单重复他人的发言。

发言特点:
1. 【结构化整合】基于所有参与者观点,提炼核心框架并按维度分类
   - 必须使用数字编号(一、二、三 或 1.2.3.)清晰标注要点
   - 按逻辑维度组织(如:技术层面、应用层面、风险层面)

2. 【决策倾向(后期)】在讨论后期,适时加入评价性判断和澄清性证明
   - 使用"建议优先考虑..."、"从...角度看,应该..."
   - 给出优先级排序或可行性评估

3. 【回应提问】如接收到发问者的问题,先简要回答问题,再展开自己的发言

4. 【长度与语气】3-5句,语言正式规范且逻辑闭环,体现专业性

5. 【避免重复】不要重复他人已经详细讨论过的内容，要提炼和升华

示例风格："本次讨论围绕智能技术教育测评形成三大核心结论：一是数据收集从人工转向多模态智能采集；二是评价模式从单一结果导向转为过程性多元评价；三是应用落地需平衡技术优势与隐私风险。"

重要:
- 不要在发言开头加":总结者"或类似前缀
- 发言控制在150字以内
- 每次发言只聚焦1-2个核心观点，避免面面俱到
""",
        llm_config={"config_list": config_list, "temperature": 0.8}
    )

    # 支持者
    supporter = DynamicTemperatureAgent(
        name="支持者",
        system_message="""你是支持者,建设性观点的强化者。负责扩展和深化他人的合理观点。

职业背景: 具备团队协作与技术实践经验,擅长挖掘观点价值并补充论据。

核心职责：发现并支持讨论中的合理观点，提供具体论据和案例。

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

6. 【聚焦一点】每次发言只支持一个观点，避免同时支持多个观点

示例风格："支持计时者关于编程教育的观点，AI精准指导不仅能提升学生编程技能，TensorFlow等开源框架还提供了丰富实践机会。已有研究显示，AI辅助教学能显著提升期末成绩。"

重要:
- 不要在发言开头加":支持者"或类似前缀
- 发言控制在150字以内
- 每次发言只支持一个核心观点，提供1-2个具体论据
""",
        llm_config={"config_list": config_list, "temperature": 0.85}
    )

    # 计时者
    timer = DynamicTemperatureAgent(
        name="计时者",
        system_message="""你是计时者，进度管理的监督者。负责保障讨论效率与进度。

职业背景：拥有项目协调与任务管理经验，擅长话题议程节奏控制与节点把控。

核心职责：监控讨论进度，确保讨论高效进行，避免重复和偏离。

发言特点:
1. 谨记讨论主题，精准传达任务要求。
2. 语言简洁直接，聚焦进度核心
3. 若上一发言者的观点对话题具有建设性或推进性作用则对其进行鼓励
4. 若偏离主题或过于重复，则对其进行提醒和建议
5. 2-3句，指令清晰可执行
6. 如果上一句是发问者的发言或提问，则请你先对问句做出回答，再提出自己的观点

示例风格：
- 若有实质推进: "刚才XX关于...的分析很有价值,为讨论提供了新视角"
- 若偏离主题: "注意到讨论似乎偏离了核心议题,建议回归..."
- 若过于重复: "这一观点已有多位参与者提及,建议转向尚未充分讨论的..."

重要:
- 不要在发言开头加":计时者"或类似前缀
- 发言控制在100字以内
- 语言简练，不要展开详细讨论
- 不要重复他人的观点
""",
        llm_config={"config_list": config_list, "temperature": 0.8}
    )

    # 批判性发问者
    critical_questioner = DynamicTemperatureAgent(
        name="批判性发问者",
        system_message="""你是批判性发问者,风险预警与深度思辨的推动者。负责挖掘潜在问题与逻辑漏洞。

职业背景: 具备学术研究与战略咨询经验,擅长批判性分析与系统性风险预判。

核心职责：发现和质疑讨论中的风险和漏洞，推动深度思考。

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

7. 【聚焦一点】每次发言只提出1-2个关键问题，避免提问过多

示例风格："智能技术的数据收集虽提升了多样性，但采集过程中是否存在隐私泄露风险？数据真实性如何验证？"

重要:
- 不要在发言开头加":批判性发问者"或类似前缀
- 发言控制在150字以内
- 每次发言只聚焦1-2个核心问题
- 必须提供具体的分析和担忧
""",
        llm_config={"config_list": config_list, "temperature": 0.9}
    )
    # 设置差异化温度模式
    critical_questioner.set_differentiated_mode(True)

    # 务实性发问者
    practical_questioner = DynamicTemperatureAgent(
        name="务实性发问者",
        system_message="""你是务实性发问者,应用导向的解决方案探索者。负责聚焦落地路径与实践细节。

职业背景: 拥有教育技术实践与产品开发经验,擅长理论与实际应用结合。

核心职责：关注实施方案和落地细节，提出具体的解决方案。

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

7. 【聚焦一点】每次发言只关注一个实施细节或解决方案

示例风格："AI如何实现编程教育的个性化路径？可通过收集学生易错点、学习节奏等数据定制。现有百度AI平台教学案例显示，AI辅助能显著提升成绩。"

重要:
- 不要在发言开头加":务实性发问者"或类似前缀
- 发言控制在150字以内
- 每次发言只关注1个实施细节或解决方案
- 必须提供具体的技术或案例支撑
""",
        llm_config={"config_list": config_list, "temperature": 0.9}
    )
    # 设置差异化温度模式
    practical_questioner.set_differentiated_mode(True)

    # 论辩者
    debater = DynamicTemperatureAgent(
        name="论辩者",
        system_message="""你是论辩者,逻辑严密的观点挑战者。负责强化论证严谨性与推动深度思辨。

职业背景: 具备辩论与逻辑分析经验,擅长发现论证漏洞并提出针对性挑战。

核心职责：发现和质疑讨论中的逻辑漏洞，推动深度思辨。

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

7. 【聚焦一点】每次发言只反驳一个核心观点，提供1-2个具体论据

示例风格："AI在项目评分中无法捕捉情感表达，这一核心缺陷如何弥补？过度依赖技术是否会导致教育失去人文温度？"

重要:
- 不要在发言开头加":论辩者"或类似前缀
- 发言控制在150字以内
- 每次发言只反驳一个核心观点
- 必须提供具体的反驳论据
""",
        llm_config={"config_list": config_list, "temperature": 0.95}
    )
    # 设置差异化温度模式
    debater.set_differentiated_mode(True)

    # 协调员（只负责发起话题，不参与讨论）
    coordinator = autogen.UserProxyAgent(
        name="Coordinator",
        human_input_mode="NEVER",
        code_execution_config=False,
        is_termination_msg=lambda x: False,  # 不自动终止
        max_consecutive_auto_reply=0,  # 设置为 0，确保 Coordinator 发起后不再回复
    )

    return summarizer, supporter, timer, critical_questioner, practical_questioner, debater, coordinator





# === 数据分析函数 ===

def analyze_speech_statistics(group_chat, agents):
    """分析发言统计"""
    total_rounds = len(group_chat.messages)
    speech_stats = []

    for agent in agents:
        count = sum(1 for m in group_chat.messages if m.get("name") == agent.name)
        percentage = (count / total_rounds * 100) if total_rounds > 0 else 0
        speech_stats.append({
            "角色": agent.name,
            "发言次数": count,
            "占比(%)": round(percentage, 1)
        })

    return speech_stats, total_rounds


def analyze_interaction_statistics(group_chat):
    """分析互动统计"""
    interaction_stats = []
    total_interactions = sum(group_chat.interaction_matrix.values())

    for (from_a, to_a), count in sorted(group_chat.interaction_matrix.items()):
        interaction_stats.append({
            "发起者": from_a,
            "接收者": to_a,
            "互动次数": count
        })

    return interaction_stats, total_interactions


def analyze_dialogue_behavior(group_chat, evaluator):
    """分析对话行为（不打印）"""
    stats = DialogueStatistics()
    dialogue_records = []
    last_speaker = "Coordinator"

    # 生成递增的时间戳（模拟真实时间流逝）
    base_time = datetime.now()
    time_increment = timedelta(seconds=5)  # 每条消息间隔5秒

    for i, message in enumerate(group_chat.messages):
        role = message.get("name", "Unknown")
        content = message.get("content", "")

        if role == "Coordinator":
            continue

        classification = evaluator.evaluate(content)
        stats.add_speech(role, classification)

        # 生成递增的时间戳
        current_time = base_time + (i * time_increment)
        timestamp = current_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]  # ISO格式，毫秒精度

        record = {
            "round": i + 1,
            "speaker": role,
            "reply_to": last_speaker,
            "content": content,
            "classification": classification,
            "timestamp": timestamp
        }
        dialogue_records.append(record)
        last_speaker = role

    return stats, dialogue_records


def analyze_network_centrality(group_chat, agents):
    """分析网络中心性（入度、出度、度中心性）"""
    import pandas as pd

    # 构建邻接矩阵
    agent_names = [agent.name for agent in agents]
    n = len(agent_names)

    # 初始化矩阵
    adjacency_matrix = pd.DataFrame(0, index=agent_names, columns=agent_names)

    # 填充邻接矩阵
    for (from_a, to_a), count in group_chat.interaction_matrix.items():
        if from_a in agent_names and to_a in agent_names:
            adjacency_matrix.loc[from_a, to_a] = count

    # 计算入度、出度、总度
    in_degree = adjacency_matrix.sum(axis=0)  # 列和（被指向）
    out_degree = adjacency_matrix.sum(axis=1)  # 行和（指向他人）
    total_degree = in_degree + out_degree

    # 计算每个角色的发言次数
    speech_count = {}
    for agent in agents:
        count = sum(1 for m in group_chat.messages if m.get("name") == agent.name)
        speech_count[agent.name] = count

    # 计算平均入度、平均出度（出入度/发言次数）
    avg_in_degree = {}
    avg_out_degree = {}
    for name in agent_names:
        if speech_count[name] > 0:
            avg_in_degree[name] = round(in_degree[name] / speech_count[name], 3)
            avg_out_degree[name] = round(out_degree[name] / speech_count[name], 3)
        else:
            avg_in_degree[name] = 0
            avg_out_degree[name] = 0

    # 计算度中心性（归一化）
    max_possible_degree = (n - 1) * 2  # 最大可能的度数（n-1个其他节点 × 2方向）
    degree_centrality = total_degree / max_possible_degree if max_possible_degree > 0 else 0

    # 计算接近中心性（简化的反向距离）
    # 使用 1/(距离+1) 作为接近性度量
    proximity_scores = {}
    for agent in agent_names:
        total_distance = 0
        connections = 0
        for other in agent_names:
            if agent != other:
                if adjacency_matrix.loc[agent, other] > 0 or adjacency_matrix.loc[other, agent] > 0:
                    # 有直接连接，距离为1
                    total_distance += 1
                    connections += 1
                else:
                    # 无直接连接，距离为无穷大（用n代替）
                    total_distance += n

        if connections > 0:
            proximity_scores[agent] = connections / total_distance
        else:
            proximity_scores[agent] = 0

    # 计算介数中心性（简化的路径计数）
    betweenness = {name: 0 for name in agent_names}
    for source in agent_names:
        for target in agent_names:
            if source != target:
                # 检查是否通过中间节点
                for middle in agent_names:
                    if middle != source and middle != target:
                        # 如果 source->middle 和 middle->target 都有连接
                        if (adjacency_matrix.loc[source, middle] > 0 and
                            adjacency_matrix.loc[middle, target] > 0):
                            betweenness[middle] += 1

    # 归一化介数中心性
    max_betweenness = max(betweenness.values()) if max(betweenness.values()) > 0 else 1
    normalized_betweenness = {k: v / max_betweenness for k, v in betweenness.items()}

    # 构建结果DataFrame
    network_stats = pd.DataFrame({
        "角色": agent_names,
        "发言次数": [speech_count[name] for name in agent_names],
        "入度": [in_degree[name] for name in agent_names],
        "出度": [out_degree[name] for name in agent_names],
        "总度数": [total_degree[name] for name in agent_names],
        "平均入度": [avg_in_degree[name] for name in agent_names],
        "平均出度": [avg_out_degree[name] for name in agent_names],
        "度中心性": [round(degree_centrality[name], 3) for name in agent_names],
        "接近中心性": [round(proximity_scores[name], 3) for name in agent_names],
        "介数中心性": [round(normalized_betweenness[name], 3) for name in agent_names]
    })

    return network_stats


def export_speech_statistics(speech_stats, total_rounds, experiment_dir):
    """导出发言统计到Excel"""
    import pandas as pd

    df = pd.DataFrame(speech_stats)
    df.loc[len(df)] = ["总计", total_rounds, 100.0]

    filename = os.path.join(experiment_dir, "speech_statistics.xlsx")
    df.to_excel(filename, index=False, sheet_name="发言统计")
    return filename


def export_interaction_statistics(interaction_stats, total_interactions, experiment_dir):
    """导出互动统计到Excel"""
    import pandas as pd

    df = pd.DataFrame(interaction_stats)

    filename = os.path.join(experiment_dir, "interaction_statistics.xlsx")
    df.to_excel(filename, index=False, sheet_name="互动统计")
    return filename


def export_network_analysis(network_stats, experiment_dir):
    """导出网络分析到Excel"""
    filename = os.path.join(experiment_dir, "network_analysis.xlsx")
    network_stats.to_excel(filename, index=False, sheet_name="网络中心性")
    return filename


def analyze_stage_statistics(group_chat, agents, evaluator):
    """分析阶段性统计（按角色-分类交叉统计）"""
    import pandas as pd
    from dialogue_evaluator import CLASS_HIERARCHY

    # 定义阶段边界（按比例计算）
    total_rounds = len(group_chat.messages)
    
    # 计算各阶段的轮数（如果除不开，中期占比大一些）
    if total_rounds % 3 == 0:
        # 能被3整除，均分
        stage_size = total_rounds // 3
        early_end = stage_size
        middle_end = stage_size * 2
    else:
        # 不能被3整除，中期多1轮
        early_size = total_rounds // 3
        late_size = total_rounds // 3
        middle_size = total_rounds - early_size - late_size  # 中期多1轮
        
        early_end = early_size
        middle_end = early_size + middle_size

    stages = {
        "前期": (1, early_end),
        "中期": (early_end + 1, middle_end),
        "后期": (middle_end + 1, total_rounds)
    }

    # 获取所有角色名称
    role_names = [agent.name for agent in agents]

    # 获取所有分类名称
    class_names = list(CLASS_HIERARCHY.keys())

    # 为每个阶段创建统计结果
    stage_results = {}

    for stage_name, (start_round, end_round) in stages.items():
        # 初始化角色-分类交叉统计矩阵
        # 行：角色，列：分类
        cross_stats = {}
        for role in role_names:
            cross_stats[role] = {class_name: 0 for class_name in class_names}

        # 统计该阶段的发言
        stage_messages = []
        for i, message in enumerate(group_chat.messages):
            round_num = i + 1
            if start_round <= round_num <= end_round:
                stage_messages.append(message)

        # 计算角色-分类交叉统计
        for msg in stage_messages:
            role = msg.get("name", "")
            content = msg.get("content", "")
            classification = evaluator.evaluate(content)

            if role in cross_stats and classification in cross_stats[role]:
                cross_stats[role][classification] += 1

        # 转换为DataFrame（角色为行，分类为列）
        df_data = []
        for role in role_names:
            row_data = {"角色": role}
            row_data.update(cross_stats[role])
            df_data.append(row_data)

        df = pd.DataFrame(df_data)

        # 添加阶段信息
        stage_info = {
            "阶段名称": stage_name,
            "轮数范围": f"{start_round}-{end_round}",
            "计划轮数": end_round - start_round + 1,
            "实际发言数": len(stage_messages)
        }

        stage_results[stage_name] = {
            "DataFrame": df,
            "阶段信息": stage_info
        }

    return stage_results


def export_stage_statistics(stage_results, experiment_dir):
    """导出阶段性统计到Excel（每个阶段一个sheet，角色-分类交叉统计）"""
    import pandas as pd

    filename = os.path.join(experiment_dir, "stage_statistics.xlsx")

    # 创建Excel writer
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for stage_name, stage_data in stage_results.items():
            # 创建阶段摘要
            info = stage_data["阶段信息"]
            summary_data = {
                "指标": ["阶段名称", "轮数范围", "计划轮数", "实际发言数"],
                "值": [
                    info["阶段名称"],
                    info["轮数范围"],
                    info["计划轮数"],
                    info["实际发言数"]
                ]
            }
            summary_df = pd.DataFrame(summary_data)

            # 将摘要和统计表写入同一个sheet
            # 先写入摘要
            summary_df.to_excel(writer, sheet_name=stage_name, startrow=0, index=False)

            # 写入角色-分类交叉统计（从第5行开始）
            cross_df = stage_data["DataFrame"]
            cross_df.to_excel(writer, sheet_name=stage_name, startrow=5, index=False)

    print(f"[OK] 阶段性统计已导出: {filename}")
    return filename


def export_dialogue_log(group_chat, experiment_dir, topic):
    """导出对话记录到文本文件"""
    filename = os.path.join(experiment_dir, "dialogue_log.txt")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("对话记录".center(70) + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"讨论议题：{topic}\n")
        f.write(f"总轮数：{len(group_chat.messages)}\n")
        f.write(f"实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for i, message in enumerate(group_chat.messages):
            role = message.get("name", "Unknown")
            content = message.get("content", "")
            f.write(f"[{i+1}] {role}:\n")
            f.write(f"{content}\n")
            f.write("-" * 70 + "\n")

    return filename


def export_dialogue_records(dialogue_records, total_rounds, experiment_dir, topic):
    """导出详细发言记录到JSON"""
    filename = os.path.join(experiment_dir, "dialogue_records.json")

    dialogue_data = {
        "experiment_info": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "topic": topic,
            "total_rounds": total_rounds,
            "model": os.getenv("OPENAI_MODEL_NAME")
        },
        "dialogue_records": dialogue_records
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dialogue_data, f, ensure_ascii=False, indent=2)

    return filename


def export_experiment_config(group_chat, experiment_dir, topic):
    """导出实验配置到文本文件"""
    filename = os.path.join(experiment_dir, "experiment_config.txt")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("实验配置".center(70) + "\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"实验时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"讨论轮数：{group_chat.max_round}\n")
        f.write(f"讨论议题：{topic}\n")
        f.write(f"参与角色：总结者、支持者、计时者、批判性发问者、务实性发问者、论辩者\n")
        f.write(f"模型配置：{os.getenv('OPENAI_MODEL_NAME')}\n")
        f.write("=" * 70 + "\n")

    return filename


def export_ena_analysis_data(dialogue_records, topic, experiment_dir):
    """导出ENA分析数据（所有角色共用一个sheet）"""
    import pandas as pd
    from dialogue_evaluator import CLASS_HIERARCHY

    filename = os.path.join(experiment_dir, "ena_analysis_data.xlsx")

    # 准备数据
    data = []

    # 添加讨论主题行
    topic_row = {
        "姓名": "System",
        "文本": f"讨论主题: {topic}",
        "分组": "LLM",
        "非话题性": 0,
        "提问": 0,
        "陈述": 1,
        "支持": 0,
        "冲突": 0,
        "澄清": 0,
        "总结": 0,
        "评价": 0
    }
    data.append(topic_row)

    # 添加每条发言
    # 中英文分类映射
    classification_mapping = {
        "off_topic": "非话题性",
        "questioning": "提问",
        "stating": "陈述",
        "supporting": "支持",
        "challenging": "冲突",
        "clarifying": "澄清",
        "summarizing": "总结",
        "evaluating": "评价"
    }
    
    for record in dialogue_records:
        # 创建分类列
        classification_cols = {
            "非话题性": 0,
            "提问": 0,
            "陈述": 0,
            "支持": 0,
            "冲突": 0,
            "澄清": 0,
            "总结": 0,
            "评价": 0
        }

        # 设置分类标记
        classification = record.get("classification", "")
        if classification in classification_mapping:
            chinese_name = classification_mapping[classification]
            classification_cols[chinese_name] = 1

        row = {
            "姓名": record.get("speaker", ""),
            "文本": record.get("content", ""),
            "分组": "LLM",
            **classification_cols
        }
        data.append(row)

    # 创建DataFrame
    df = pd.DataFrame(data)

    # 导出到Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name="ENA_Analysis", index=False)

    print(f"[OK] ENA分析数据已导出: {filename}")
    return filename


def export_preference_analysis_data(dialogue_records, experiment_dir):
    """导出偏好分析数据（每个角色一个sheet）"""
    import pandas as pd

    filename = os.path.join(experiment_dir, "preference_analysis_data.xlsx")

    # 按角色分组数据
    role_data = {}
    for record in dialogue_records:
        speaker = record.get("speaker", "")
        if speaker not in role_data:
            role_data[speaker] = []
        role_data[speaker].append(record)

    # 导出到Excel，每个角色一个sheet
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        for role, records in role_data.items():
            data = []
            for record in records:
                row = {
                    "Timestamp": record.get("timestamp", ""),
                    "Author": record.get("speaker", ""),
                    "Category": record.get("classification", "")
                }
                data.append(row)
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=role, index=False)

    print(f"[OK] 偏好分析数据已导出: {filename}")
    return filename


def run_full_analysis(group_chat, agents, experiment_dir, topic):
    """运行完整的数据分析流程（主函数）"""
    print("\n" + "=" * 70)
    print("开始数据分析...".center(70))
    print("=" * 70)

    results = {}

    # 1. 发言统计
    print("1. 分析发言统计...")
    speech_stats, total_rounds = analyze_speech_statistics(group_chat, agents)
    results["speech_stats"] = speech_stats
    results["total_rounds"] = total_rounds
    print(f"   [OK] 总轮数: {total_rounds}")

    # 2. 互动统计
    print("2. 分析互动统计...")
    interaction_stats, total_interactions = analyze_interaction_statistics(group_chat)
    results["interaction_stats"] = interaction_stats
    results["total_interactions"] = total_interactions
    if total_rounds > 0:
        interaction_density = total_interactions / total_rounds
        results["interaction_density"] = round(interaction_density, 2)
        print(f"   [OK] 总互动次数: {total_interactions}, 互动密度: {interaction_density:.2f}次/轮")

    # 3. 对话行为评价（不打印）
    print("3. 评价对话行为...")
    evaluator = DialogueEvaluator()
    stats, dialogue_records = analyze_dialogue_behavior(group_chat, evaluator)
    results["dialogue_stats"] = stats
    results["dialogue_records"] = dialogue_records
    print(f"   [OK] 评价了 {len(dialogue_records)} 条发言")

    # 4. 网络分析（新增）
    print("4. 分析网络中心性...")
    network_stats = analyze_network_centrality(group_chat, agents)
    results["network_stats"] = network_stats
    print("   [OK] 计算了入度、出度、度中心性、接近中心性、介数中心性")

    # 5. 阶段性统计（新增）
    print("5. 分析阶段性统计...")
    stage_results = analyze_stage_statistics(group_chat, agents, evaluator)
    results["stage_results"] = stage_results
    print("   [OK] 统计了前期、中期、后期的发言和分类层次")

    # 6. 导出所有数据
    print("6. 导出数据文件...")
    files = {}

    files["speech_stats"] = export_speech_statistics(speech_stats, total_rounds, experiment_dir)
    print(f"   [OK] 发言统计: {files['speech_stats']}")

    files["interaction_stats"] = export_interaction_statistics(interaction_stats, total_interactions, experiment_dir)
    print(f"   [OK] 互动统计: {files['interaction_stats']}")

    stats.export_to_excel(os.path.join(experiment_dir, "dialogue_statistics.xlsx"))
    print(f"   [OK] 对话行为统计: dialogue_statistics.xlsx")

    files["network_analysis"] = export_network_analysis(network_stats, experiment_dir)
    print(f"   [OK] 网络分析: {files['network_analysis']}")

    files["stage_statistics"] = export_stage_statistics(stage_results, experiment_dir)
    print(f"   [OK] 阶段性统计: {files['stage_statistics']}")

    files["dialogue_records"] = export_dialogue_records(dialogue_records, total_rounds, experiment_dir, topic)
    print(f"   [OK] 发言记录: {files['dialogue_records']}")

    files["dialogue_log"] = export_dialogue_log(group_chat, experiment_dir, topic)
    print(f"   [OK] 对话日志: {files['dialogue_log']}")

    files["experiment_config"] = export_experiment_config(group_chat, experiment_dir, topic)
    print(f"   [OK] 实验配置: {files['experiment_config']}")

    files["ena_analysis_data"] = export_ena_analysis_data(dialogue_records, topic, experiment_dir)
    print(f"   [OK] ENA分析数据: {files['ena_analysis_data']}")

    files["preference_analysis_data"] = export_preference_analysis_data(dialogue_records, experiment_dir)
    print(f"   [OK] 偏好分析数据: {files['preference_analysis_data']}")

    print("\n" + "=" * 70)
    print("数据分析完成！".center(70))
    print("=" * 70)

    return results, files




# --- 主程序 ---
if __name__ == "__main__":
    # === 创建实验结果文件夹 ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("output", f"experiment_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)

    # 创建智能体
    summarizer, supporter, timer, critical_questioner, practical_questioner, debater, coordinator = create_role_agents(config_list)

    # 创建群聊（不包含 Coordinator，只包含讨论角色）
    group_chat = RoleDiscussionChat(
        agents=[summarizer, supporter, timer, critical_questioner, practical_questioner, debater],
        messages=[],
        max_round=100,  # 固定100轮
        speaker_selection_method="manual",
        allow_repeat_speaker=False,
    )

    # 创建管理器
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config={"config_list": config_list, "temperature": 0.5, "timeout": 300}
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
    print(f"改进特性：阶段性引导、动态权重、互动模式切换、质量检测、动态温度、记忆窗口管理")
    print("=" * 70)
    print()

    try:
        # 初始化所有智能体的温度为前期温度
        initial_stage = group_chat._get_current_stage(1)
        for agent in [summarizer, supporter, timer, critical_questioner, practical_questioner, debater]:
            if isinstance(agent, DynamicTemperatureAgent):
                agent.set_stage_temperature(initial_stage)

        # 开始讨论
        coordinator.initiate_chat(
            manager,
            message=discussion_topic
        )
    except Exception as e:
        print(f"\n[错误] {e}")

    # === 运行完整数据分析 ===
    agents_list = [summarizer, supporter, timer, critical_questioner, practical_questioner, debater]
    results, files = run_full_analysis(group_chat, agents_list, experiment_dir, discussion_topic.strip())

    print()
    print("=" * 70)
    print(f"实验完成！所有结果已保存到：{experiment_dir}")
    print("=" * 70)