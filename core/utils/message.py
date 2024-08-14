# -*- coding:utf-8 -*-
# Create with PyCharm.
# @Project Name: CUSTrack
# @Author      : Shukang Zhang  
# @Owner       : amax
# @Data        : 2024/7/31
# @Time        : 21:39
# @Description :
import os
def push(title, content):
    import requests
    import datetime
    with open(os.path.expanduser('~/.push_key')) as f:
        key = f.read()
        base_url = (f'https://api2.pushdeer.com/message/push?pushkey={key}&'
                    f'text={title}&'
                    f'desp={content}\n' + f'> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}&'
                                          f'type=markdown')
        return requests.get(base_url)