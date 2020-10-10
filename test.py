#! -*- coding: utf-8 -*-
# NEZHA模型做闲聊任务
# 测试脚本
# 测试环境：tensorflow 1.14 + keras 2.3.1 + bert4keras 0.8.4

import numpy as np
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import AutoRegressiveDecoder

# nezha配置
config_path = '/root/kg/bert/nezha_gpt_dialog/config.json'
checkpoint_path = '/root/kg/bert/nezha_gpt_dialog/model.ckpt'
dict_path = '/root/kg/bert/nezha_gpt_dialog/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 建立并加载模型
model = build_transformer_model(
    config_path,
    checkpoint_path,
    model='nezha',
    application='lm',
)
model.summary()


class ChatBot(AutoRegressiveDecoder):
    """基于随机采样对话机器人
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        curr_segment_ids = np.ones_like(output_ids) - segment_ids[0, -1]
        segment_ids = np.concatenate([segment_ids, curr_segment_ids], 1)
        return model.predict([token_ids, segment_ids])[:, -1]

    def response(self, texts, topk=5):
        token_ids, segment_ids = [tokenizer._token_start_id], [0]
        for i, text in enumerate(texts):
            ids = tokenizer.encode(text)[0][1:]
            token_ids.extend(ids)
            segment_ids.extend([i % 2] * len(ids))
        results = self.random_sample([token_ids, segment_ids], 1, topk)
        return tokenizer.decode(results[0])


chatbot = ChatBot(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)
print(chatbot.response([u'别爱我没结果', u'你这样会失去我的', u'失去了又能怎样']))
"""
回复是随机的，例如：那你还爱我吗 | 不知道 | 爱情是不是不能因为一点小事就否定了 | 我会一直爱你，你一个人会很辛苦 | 等等。
"""
