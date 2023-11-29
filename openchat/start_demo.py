# from openchat.openchat import OpenChat
# from demo.web_demo_env import WebDemoEnv

from openchat import OpenChat
from demo import WebDemoEnv

# OpenChat(model='vqa_model_lxmert', env=WebDemoEnv())
OpenChat(model='vqa_model_oscar', env=WebDemoEnv())
# OpenChat(model='vqa_model_vilbert', env=WebDemoEnv())
# OpenChat(model='vqa_model_vinvl', env=WebDemoEnv())
# OpenChat(model='vqa_model_devlbert', env=WebDemoEnv())
# OpenChat(model='vqa_model_uniter', env=WebDemoEnv())
