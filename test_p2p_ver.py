import autogen
from autogen import config_list_from_json

# --- 配置和 Agent 定义 (与之前相同) ---
config_list = config_list_from_json("OAI_CONFIG_LIST.json")

teacher = autogen.AssistantAgent(
    name="Teacher",
    system_message="你是一位物理老师。请向学生提问，并根据学生的回答进行追问或解释。在对话结束时，请说'辅导结束'。",
    llm_config={"config_list": config_list}
)

student_a = autogen.AssistantAgent(
    name="StudentA",
    system_message="你是一个学生，请尽力回答老师提出的关于牛顿第一定律的问题。",
    llm_config={"config_list": config_list},
    is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("辅导结束"),
)

# --- 直接开始双向对话 ---
# 注意：这里不再需要 GroupChat 和 GroupChatManager
# Teacher 作为发起者，直接与 StudentA 开始对话
teacher.initiate_chat(
    student_a,
    message="你好！我们来一对一讨论一下牛顿第一定律。你先说说你对'惯性'这个词的理解吧。",
    max_turns=3 # 限制对话轮次
)