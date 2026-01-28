"""
动态知识图谱构建与演化模块
Dynamic Knowledge Graph Construction and Evolution

该模块实现了智能体在讨论过程中实时构建和更新个人知识图谱的功能。
支持概念识别、关联建立、掌握度更新等核心功能。

依赖检查：
- re: Python 内置
- json: Python 内置
- datetime: Python 内置
- typing: Python 内置
- collections: Python 内置

无需额外安装包，使用 Python 标准库即可运行。
"""

import re
import json
import sys
from datetime import datetime
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

# 设置控制台编码为 UTF-8（Windows 兼容）
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


class Concept:
    """概念节点类：表示知识图谱中的一个概念"""
    
    def __init__(self, name: str, initial_mastery: float = 0.0):
        self.name = name
        self.mastery = max(0.0, min(1.0, initial_mastery))  # 掌握度 [0, 1]
        self.connections: Set[str] = set()  # 关联的概念集合
        self.misconceptions: Set[str] = set()  # 误解集合
        self.last_mentioned: Optional[str] = None  # 最后提及时间
        self.confidence = 0.5  # 对该概念的信心度
        self.mention_count = 0  # 提及次数
        self.examples: List[str] = []  # 举例
    
    def update_mastery(self, delta: float):
        """更新掌握度"""
        self.mastery = max(0.0, min(1.0, self.mastery + delta))
    
    def add_connection(self, concept_name: str):
        """添加关联概念"""
        self.connections.add(concept_name)
    
    def add_misconception(self, misconception: str):
        """添加误解"""
        self.misconceptions.add(misconception)
    
    def add_example(self, example: str):
        """添加举例"""
        if example not in self.examples:
            self.examples.append(example)
    
    def mention(self):
        """记录一次提及"""
        self.mention_count += 1
        self.last_mentioned = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "name": self.name,
            "mastery": self.mastery,
            "connections": list(self.connections),
            "misconceptions": list(self.misconceptions),
            "last_mentioned": self.last_mentioned,
            "confidence": self.confidence,
            "mention_count": self.mention_count,
            "examples": self.examples
        }


class KnowledgeGraph:
    """知识图谱类：管理一个智能体的完整知识结构"""
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.concepts: Dict[str, Concept] = {}  # 概念字典：概念名 -> Concept对象
        self.update_history: List[Dict] = []  # 更新历史记录
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_updated = self.created_at
    
    def add_concept(self, concept_name: str, initial_mastery: float = 0.0) -> Concept:
        """添加新概念"""
        if concept_name not in self.concepts:
            concept = Concept(concept_name, initial_mastery)
            self.concepts[concept_name] = concept
            self._log_update("add_concept", concept_name, f"添加新概念，初始掌握度: {initial_mastery}")
            return concept
        return self.concepts[concept_name]
    
    def get_concept(self, concept_name: str) -> Optional[Concept]:
        """获取概念"""
        return self.concepts.get(concept_name)
    
    def update_concept_mastery(self, concept_name: str, delta: float):
        """更新概念掌握度"""
        if concept_name in self.concepts:
            old_mastery = self.concepts[concept_name].mastery
            self.concepts[concept_name].update_mastery(delta)
            new_mastery = self.concepts[concept_name].mastery
            self._log_update("update_mastery", concept_name, 
                           f"掌握度: {old_mastery:.2f} -> {new_mastery:.2f}")
    
    def add_connection(self, concept1: str, concept2: str):
        """在两个概念之间建立连接"""
        if concept1 in self.concepts and concept2 in self.concepts:
            self.concepts[concept1].add_connection(concept2)
            self.concepts[concept2].add_connection(concept1)
            self._log_update("add_connection", f"{concept1} <-> {concept2}", 
                           "建立概念关联")
    
    def process_utterance(self, utterance: str):
        """
        处理一段发言，自动识别概念并更新知识图谱
        
        参数:
            utterance: 发言内容
        """
        # 1. 提取概念（简单实现：提取名词性短语）
        extracted_concepts = self._extract_concepts(utterance)
        
        # 2. 更新或创建概念
        for concept_name in extracted_concepts:
            concept = self.add_concept(concept_name, initial_mastery=0.3)
            concept.mention()
            
            # 根据发言内容调整掌握度
            if "我觉得" in utterance or "我认为" in utterance:
                concept.update_mastery(0.05)
            elif "不理解" in utterance or "困惑" in utterance:
                concept.update_mastery(-0.1)
            elif "是" in utterance or "就是" in utterance:
                concept.update_mastery(0.1)
        
        # 3. 识别概念间的关系
        if len(extracted_concepts) >= 2:
            for i in range(len(extracted_concepts) - 1):
                self.add_connection(extracted_concepts[i], extracted_concepts[i + 1])
        
        # 4. 识别误解
        misconceptions = self._detect_misconceptions(utterance)
        for concept_name, misconception in misconceptions:
            if concept_name in self.concepts:
                self.concepts[concept_name].add_misconception(misconception)
                self._log_update("add_misconception", concept_name, 
                               f"检测到误解: {misconception}")
        
        self.last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        从文本中提取概念（简单实现）
        
        实际应用中可以使用更高级的 NLP 技术，如：
        - 命名实体识别（NER）
        - 关键词提取（TF-IDF, TextRank）
        - 领域本体匹配
        """
        concepts = []
        
        # 物理学科常见概念列表（可扩展）
        physics_concepts = [
            "牛顿第一定律", "牛顿第二定律", "牛顿第三定律",
            "惯性", "摩擦力", "重力", "弹力", "支持力",
            "匀速运动", "变速运动", "加速度", "速度", "位移",
            "质量", "力", "平衡", "自由落体", "动量", "能量",
            "动能", "势能", "功", "功率"
        ]
        
        # 检查文本中是否包含已知概念
        for concept in physics_concepts:
            if concept in text:
                concepts.append(concept)
        
        # 提取可能的名词短语（简单正则）
        # 匹配模式：形容词/名词 + 名词
        noun_phrases = re.findall(r'([a-zA-Z\u4e00-\u9fa5]{2,6})(?:是|就是|指|表示|的)', text)
        concepts.extend(noun_phrases)
        
        # 去重并过滤
        seen = set()
        unique_concepts = []
        for concept in concepts:
            if concept not in seen and len(concept) >= 2:
                seen.add(concept)
                unique_concepts.append(concept)
        
        return unique_concepts
    
    def _detect_misconceptions(self, text: str) -> List[Tuple[str, str]]:
        """
        检测文本中的误解
        
        返回: [(概念名, 误解内容), ...]
        """
        misconceptions = []
        
        # 常见误解模式
        misconception_patterns = [
            (r'(惯性)\s*(?:是|就是)\s*(?:一种|一个)?\s*力', "惯性是力"),
            (r'(摩擦力)\s*(?:总是|一定)\s*(?:阻碍|阻止)\s*(?:运动)', "摩擦力总是阻碍运动"),
            (r'(速度)\s*(?:越大|越快)\s*(?:力|受力)\s*(?:越大|越大)', "速度越大力越大"),
            (r'(重力)\s*(?:等于|就是)\s*(?:质量|重量)', "重力等于质量"),
        ]
        
        for pattern, misconception in misconception_patterns:
            match = re.search(pattern, text)
            if match:
                concept_name = match.group(1)
                misconceptions.append((concept_name, misconception))
        
        return misconceptions
    
    def _log_update(self, action_type: str, target: str, description: str):
        """记录更新日志"""
        log_entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "action": action_type,
            "target": target,
            "description": description
        }
        self.update_history.append(log_entry)
    
    def get_knowledge_summary(self) -> Dict:
        """获取知识图谱摘要"""
        if not self.concepts:
            return {"message": "知识图谱为空"}
        
        total_concepts = len(self.concepts)
        avg_mastery = sum(c.mastery for c in self.concepts.values()) / total_concepts
        total_connections = sum(len(c.connections) for c in self.concepts.values())
        
        # 找出掌握度最高和最低的概念
        sorted_by_mastery = sorted(self.concepts.items(), key=lambda x: x[1].mastery)
        weakest = sorted_by_mastery[0] if sorted_by_mastery else None
        strongest = sorted_by_mastery[-1] if sorted_by_mastery else None
        
        return {
            "agent_name": self.agent_name,
            "total_concepts": total_concepts,
            "average_mastery": round(avg_mastery, 3),
            "total_connections": total_connections,
            "weakest_concept": {
                "name": weakest[0],
                "mastery": weakest[1].mastery
            } if weakest else None,
            "strongest_concept": {
                "name": strongest[0],
                "mastery": strongest[1].mastery
            } if strongest else None,
            "last_updated": self.last_updated
        }
    
    def get_learning_gaps(self, threshold: float = 0.5) -> List[Dict]:
        """
        获取知识缺口（掌握度低于阈值的概念）
        
        参数:
            threshold: 掌握度阈值
        
        返回: 需要加强学习的概念列表
        """
        gaps = []
        for concept_name, concept in self.concepts.items():
            if concept.mastery < threshold:
                gaps.append({
                    "concept": concept_name,
                    "mastery": concept.mastery,
                    "last_mentioned": concept.last_mentioned,
                    "related_concepts": list(concept.connections),
                    "misconceptions": list(concept.misconceptions)
                })
        
        # 按掌握度排序（从低到高）
        gaps.sort(key=lambda x: x["mastery"])
        return gaps
    
    def visualize_graph(self) -> str:
        """
        生成知识图谱的可视化文本表示
        
        返回: 图谱的文本表示
        """
        if not self.concepts:
            return "知识图谱为空"
        
        lines = []
        lines.append("=" * 60)
        lines.append(f"{self.agent_name} 的知识图谱".center(60))
        lines.append("=" * 60)
        lines.append("")
        
        # 按掌握度排序显示概念
        sorted_concepts = sorted(self.concepts.items(), 
                                key=lambda x: x[1].mastery, 
                                reverse=True)
        
        for concept_name, concept in sorted_concepts:
            mastery_bar = "█" * int(concept.mastery * 20)
            lines.append(f"📚 {concept_name}")
            lines.append(f"   掌握度: {concept.mastery:.2f} [{mastery_bar:20s}]")
            lines.append(f"   信心度: {concept.confidence:.2f}")
            lines.append(f"   提及次数: {concept.mention_count}")
            
            if concept.connections:
                lines.append(f"   关联概念: {', '.join(concept.connections)}")
            
            if concept.misconceptions:
                lines.append(f"   ⚠️  误解: {', '.join(concept.misconceptions)}")
            
            if concept.examples:
                lines.append(f"   举例: {', '.join(concept.examples[:2])}")
            
            lines.append("")
        
        lines.append("=" * 60)
        return "\n".join(lines)
    
    def export_to_json(self, filepath: str):
        """导出知识图谱到 JSON 文件"""
        data = {
            "agent_name": self.agent_name,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "concepts": {name: concept.to_dict() for name, concept in self.concepts.items()},
            "update_history": self.update_history
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def import_from_json(self, filepath: str):
        """从 JSON 文件导入知识图谱"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.agent_name = data["agent_name"]
        self.created_at = data["created_at"]
        self.last_updated = data["last_updated"]
        self.update_history = data["update_history"]
        
        self.concepts = {}
        for name, concept_data in data["concepts"].items():
            concept = Concept(name, concept_data["mastery"])
            concept.connections = set(concept_data["connections"])
            concept.misconceptions = set(concept_data["misconceptions"])
            concept.last_mentioned = concept_data["last_mentioned"]
            concept.confidence = concept_data["confidence"]
            concept.mention_count = concept_data["mention_count"]
            concept.examples = concept_data["examples"]
            self.concepts[name] = concept


# ============================================================================
# 示例使用和测试
# ============================================================================

def test_knowledge_graph():
    """测试知识图谱功能"""
    print("=" * 70)
    print("动态知识图谱测试".center(70))
    print("=" * 70)
    print()
    
    # 创建学生 A 的知识图谱
    student_a_kg = KnowledgeGraph("StudentA")
    
    # 模拟讨论过程
    utterances = [
        "我觉得牛顿第一定律就是物体如果不受外力作用，它会保持静止或者匀速直线运动。",
        "这个好像是惯性吧？惯性是物体保持原来状态的性质。",
        "滑冰的时候，冰面光滑，摩擦力小，所以你可以滑得很远。",
        "如果地面有摩擦力的话，滑的距离会变短吗？我不太确定呢。",
        "我觉得惯性是力，它能让物体保持运动。",  # 这里有一个误解
        "牛顿第一定律和惯性是相关的，惯性是牛顿第一定律的核心概念。",
        "摩擦力会抵消滑冰时的动能，让你逐渐减速。",
    ]
    
    print("📝 处理讨论发言...")
    print("-" * 70)
    for i, utterance in enumerate(utterances, 1):
        print(f"发言 {i}: {utterance}")
        student_a_kg.process_utterance(utterance)
        print()
    
    # 显示知识图谱摘要
    print("\n📊 知识图谱摘要:")
    print("-" * 70)
    summary = student_a_kg.get_knowledge_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print()
    
    # 显示知识缺口
    print("\n⚠️  知识缺口（需要加强的概念）:")
    print("-" * 70)
    gaps = student_a_kg.get_learning_gaps(threshold=0.5)
    if gaps:
        for gap in gaps:
            print(f"• {gap['concept']}: 掌握度 {gap['mastery']:.2f}")
            if gap['misconceptions']:
                print(f"  误解: {', '.join(gap['misconceptions'])}")
    else:
        print("✅ 暂无明显知识缺口")
    print()
    
    # 可视化知识图谱
    print("\n🎨 知识图谱可视化:")
    print("-" * 70)
    print(student_a_kg.visualize_graph())
    
    # 导出为 JSON
    output_file = "student_a_knowledge_graph.json"
    student_a_kg.export_to_json(output_file)
    print(f"\n💾 知识图谱已导出到: {output_file}")
    
    print("\n" + "=" * 70)
    print("测试完成！".center(70))
    print("=" * 70)


def test_multi_agent_comparison():
    """测试多智能体知识图谱对比"""
    print("\n\n" + "=" * 70)
    print("多智能体知识图谱对比测试".center(70))
    print("=" * 70)
    print()
    
    # 创建三个学生的知识图谱
    students = {
        "StudentA": KnowledgeGraph("StudentA"),
        "StudentB": KnowledgeGraph("StudentB"),
        "StudentC": KnowledgeGraph("StudentC")
    }
    
    # 模拟不同学生的发言
    student_utterances = {
        "StudentA": [
            "我觉得牛顿第一定律讲的是物体不受力会保持原状态。",
            "惯性就是物体保持运动状态的性质吧？",
            "滑冰的时候摩擦力小，所以能滑很远。"
        ],
        "StudentB": [
            "牛顿第一定律也叫惯性定律，是力学的基础。",
            "惯性是物体的固有属性，和质量有关，与运动状态无关。",
            "摩擦力会阻碍相对运动，但不一定阻碍运动。"
        ],
        "StudentC": [
            "牛顿第一定律在实际中很难实现，因为总有摩擦力。",
            "但是这个定律是理想化的，帮助我们理解运动规律。",
            "我觉得惯性不是力，而是一种性质。"
        ]
    }
    
    # 处理每个学生的发言
    for student_name, utterances in student_utterances.items():
        print(f"📝 处理 {student_name} 的发言...")
        for utterance in utterances:
            students[student_name].process_utterance(utterance)
        print()
    
    # 对比分析
    print("\n📊 多智能体知识对比:")
    print("-" * 70)
    
    comparison = []
    for student_name, kg in students.items():
        summary = kg.get_knowledge_summary()
        comparison.append({
            "student": student_name,
            "concepts": summary["total_concepts"],
            "avg_mastery": summary["average_mastery"],
            "connections": summary["total_connections"]
        })
    
    # 打印对比表格
    print(f"{'学生':<12} {'概念数':<10} {'平均掌握度':<12} {'连接数':<10}")
    print("-" * 70)
    for comp in comparison:
        print(f"{comp['student']:<12} {comp['concepts']:<10} "
              f"{comp['avg_mastery']:<12.3f} {comp['connections']:<10}")
    
    # 找出掌握度最高的概念（每个学生）
    print("\n🏆 各学生掌握度最高的概念:")
    print("-" * 70)
    for student_name, kg in students.items():
        if kg.concepts:
            best_concept = max(kg.concepts.items(), key=lambda x: x[1].mastery)
            print(f"{student_name}: {best_concept[0]} (掌握度: {best_concept[1].mastery:.2f})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    # 运行测试
    test_knowledge_graph()
    test_multi_agent_comparison()