from LAC import LAC

# 装载LAC模型
lac = LAC(mode='lac')

# # 单个样本输入，输入为Unicode编码的字符串
# text = u"LAC是个优秀的分词工具"
# lac_result = lac.run(text)

# # 批量样本输入, 输入为多个句子组成的list，平均速率更快
# texts = ["当时，在北京市市场监督管理局官网公布的电动车抽检信息中，天津爱玛车业科技有限公司生产的5批次电动自行车存在不合格问题，包括最高车速、脚踏行驶能力、整车质量、反射器和鸣号装置、欠压、过流保护功能等问题",'电动自行车']
# texts = ["中国医药：子公司两批次药品不符合规定 被药监局罚没457.13万元"]
# lac_result = lac.run(texts)
# print(lac_result)

import json

news = []
with open('output/DuEE1.0/duee_surbot.json', 'r', encoding='utf-8') as fr:
    for new in fr.readlines():
        news.append(json.loads(new))
for new in news: # 针对每一个json
    if len(new['event_list']): # 含有事件
        text = lac.run(new['text']) 
        print(text)
        for event in new['event_list']: # 针对每一个时间
            arguments = []
            for role in event['arguments']: # 针对每一个时间的arguments
                arguments.append(role['argument'])
            arguments_fenci = lac.run(arguments)
            for index_argument in range(len(arguments_fenci)): # 针对每一个argument
                for index_fenci in range(len(arguments_fenci[index_argument][0])): # 针对每一个argument的每一个分词结果
                    res = arguments_fenci[index_argument][0][index_fenci] # 每一个分词结果
                    for index_text_fenci in range(len(text[0])):
                        if res in text[0][index_text_fenci]:
                            if text[1][index_text_fenci] != 'v':
                                arguments_fenci[index_argument][0][index_fenci] = text[0][index_text_fenci]
                            else:
                                pass
            for index_argument in range(len(arguments_fenci)): # 针对每一个argument
                argument = ''.join(arguments_fenci[index_argument][0])
                # argument = ''.join(sorted(arguments_fenci[index_argument][0]), key=arguments_fenci[index_argument][0].index) # 去重
                event['arguments'][index_argument]['argument'] = argument            

with open('output/DuEE1.0/duee_surbot_fenci.json', 'w', encoding='utf-8') as fw:
    for new in news:
        fw.write(json.dumps(new, ensure_ascii=False))
        fw.write('\n')
