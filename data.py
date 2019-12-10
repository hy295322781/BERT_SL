# -*- coding:utf-8 -*-
"""
Created on 2019-11-15 09:37:58
Author: Xiong Zecheng (295322781@qq.com)
"""
import json
import random
import codecs
import string

def read_ccks_train(filename):
    '''
    读取CCKS2018训练数据
    '''
    data = []
    with open(filename,encoding="UTF-8") as f:
        for line in f.readlines():
            line = line.replace('"',"")
            string, entitys = line.strip().split('{')
            string = string.split(',')[-2]
            entitys = entitys.replace('}',"")
            entitys = entitys.replace('，',",")
            entitys = entitys.replace(';',",")
            if entitys.find('lyrics') != -1:
                continue
            string = string.lower()
            string = string.replace("儿童歌曲","儿歌")
            string = string.split("\t")[0]
            entitys = entitys.split(",")
            entity_list = []
            for entity in entitys:
                pairs = entity.split(":")
                if len(pairs)!=2:
                    continue
                tag,name = pairs
                tag = tag.strip()
                name = name.strip().lower()
                if tag == "artist":
                    tag = "Actor"
                elif tag == "song":
                    tag = "Song"
                elif tag == "album":
                    tag = "Album"
                elif tag in ['language', 'genre', 'tag', 'scene', 'mood']:
                    tag = "Label"
                else:
                    continue
                position = string.find(name)
                if position == -1:
                    continue
                entity_list.append((name,tag))
            data.append(
                {
                    "sentence": string,
                    "entity": entity_list
                }
            )
    return data

def write_segment_dict():
    record = list()
    with open('data/music_base/actor.ch.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            record.append(line.strip())
    with open('data/music_base/song.ch.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            record.append(line.strip())
    with open('data/music_base/album.ch.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            record.append(line.strip())
    with open('data/music_base/movie.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            record.append(line.strip())

    skip_char = set("，。？！听唱的我"+string.printable)
    i = 0
    while i<len(record):
        if skip_char&set(record[i]):
            record.pop(i)
        else:
            i += 1

    with open('data/user_cws_lexicon.txt','w',encoding="UTF-8") as f:
        f.write("\n".join(record))

def write_data(data_num,istrain=True):
    # scope - 歌, 专辑
    # undef - <song>, <album>, <label>, <actor>都有可能
    # person - <actor>, 指示代词
    # thing - 张专辑、专辑、专辑中、张专辑中、专辑里

    actor_list_en = []
    actor_list_ch = []
    song_list_en = []
    song_list_ch = []
    album_list_en = []
    album_list_ch = []
    movie_list = []
    with open('data/music_base/actor.en.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            actor_list_en.append(line.strip())
    with open('data/music_base/song.en.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            song_list_en.append(line.strip())
    with open('data/music_base/actor.ch.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            actor_list_ch.append(line.strip())
    with open('data/music_base/song.ch.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            song_list_ch.append(line.strip())
    with open('data/music_base/album.ch.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            album_list_ch.append(line.strip())
    with open('data/music_base/album.en.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            album_list_en.append(line.strip())
    with open('data/music_base/movie.dict', 'r',encoding="UTF-8") as f:
        for line in f.readlines():
            movie_list.append(line.strip())

    # 标签
    fengge_adj = ["摇滚","古典","爵士","电子","R&B","古风","乡村","嘻哈","拉丁","雷鬼","新世纪","中国风","男声","女声"]
    fengge_n = ["纯音乐","民谣","民歌","舞曲","说唱","戏曲","轻音乐","歌剧","蓝调"]
    changjing = ["校园","雨天","开车","旅行","运动","咖啡厅","毕业","夜店","工作","睡前","派对","熬夜","学习","熬夜","午后","购物","坐车"]
    qinggan = ["伤感","快乐","思念","寂寞","失恋","甜蜜","祝福","感动","抒情","恋爱","安静","温暖","激情","怀念","虐心","轻松","愤怒","怀旧","悠扬","搞笑","宣泄","情侣"]
    zhuti_adj = ["经典","分手","军旅","友情","KTV","亲情","背景","佛教","武侠","网络","胎教","方言"]
    zhuti_n = ["铃声","红歌","儿歌","成名曲"]
    yueqi = ["钢琴","吉他","古筝","小提琴","琵琶","二胡","萨克斯","葫芦丝","古琴"]
    tezhi = ["青春","故乡","励志","草原","治愈","文艺","清新","异域","神曲","雷人","独自"]
    yuyan = ["国语","华语","粤语","英语","英文","闽南语","俄语","外语","中文","潮汕话","泰国","港台","普通话"]
    niandai = [str(i) + "0年代" for i in range(0,10)]
    # 形容词类型标签
    label1 = qinggan + tezhi + fengge_adj + zhuti_adj

    # 名词类型标签
    label2 = fengge_n + zhuti_n

    # 场景情景类标签
    label3 = changjing + qinggan

    # 可以成为歌曲类型前缀的标签
    label4 = fengge_adj + yueqi + yuyan

    # 年代+语言，可以修饰专辑的标签
    label5 = niandai + yuyan

    # 可以修饰音乐的标签
    label6 = yueqi + ["纯音乐","现场","轻音乐"] + yuyan

    # 句子中的相应片段
    c_start = ["请给我播", "播放", "播", "有", "有没有", "请播一首", "我想听", "想听", "能播", "可以播",
               "给我播放", "给我播", "能不能播","搜索", "再播", "查", "查询", "给我播放", "放", "找", "搜", "找一下",
               "切成", "切换成", "其它", "别的", "那就","我要","要", "帮我找", "帮我查一下",
               "我想要", "我想要听", "我要听", "找出", "帮我找出", "帮我搜索", "能不能搜索", "能不能找出"]
    c_end = ["音乐","歌曲","歌"]*2+["曲子","乐曲"]
    q_start = ["请问","问下","请问下","请问一下","我想问下","我想请问","我想请问一下","我请问一下","我问下","我请问下"]
    q_piece = ["还有不有","有没有","有没","还有没","还有没有"]
    q_end = ["有吗","有没有","有哪些","有没","有没啊","有哪些啊","有没有啊","有不","吗"]
    num_piece = ["来一首", "一首", "来几首", "几首", "再来一首", "来一个", "找一首", "放一首", "播一首", "来首",
                 "搞首", "来个", "搞几首", "搞几个", "再来几首", "来几个", "找几首", "放几首", "播几首", "推荐几首",
                 "推荐一首", "推送几首", "推送一首","唱一首","唱几首"]
    search = ["搜索", "查", "查询", "找", "搜", "查一下", "推荐", "推送", "查找", "推", "找一下", "查找一下",
              "搜一下", "搜索一下", "寻找"]
    other = ["其他","其他的","别的","另外的","另一个","更多","更多的"]

    before_label_1 = ["比较", "有点", "稍微", "非常", "很", "有点儿", "有一些", "感觉", "听起来", "感觉很",
                      "感觉有一些", "感觉有点", "感觉有点儿", "听上去","听上去有些","充满","让人","让人感觉"]
    before_label_3 = ["适合","关于","在"]
    after_label_1 = ["一点","风格","一些"]
    after_label_3 = ["的时候听的","时听的","的时候放的","时放的","听的","放的"]
    after_label_4 = ["曲","乐","歌","歌曲","乐曲"]

    # 问歌手，重复<actor>，是因为一般问一个歌手的频率比较高，让问合唱的与一个的频率相同
    a_actor = ["<actor>和<actor>", "<actor>或者<actor>", "<actor>、<actor>", "<actor>和<actor>", "<actor>与<actor>",
               "<actor>，<actor>", "<actor>跟<actor>"]+["<actor>"]*35
    # 歌手与歌的片段
    a_m_song = ["合唱的","演唱的","唱的","唱过的","一起唱的", "版本的", "演唱会的","音乐会的","演奏的"]+["的"]*6
    a_e_song = ["<song>"]*15+c_end
    after_song = ["这首歌，这个歌曲，这首歌曲，这一首，这一首歌，该首歌，此首歌，此首歌曲，该首歌曲，这一首歌曲"]

    # 专辑相关片段
    before_album = ["这", "那", "此", "这张", "那张", "此张"]
    album_piece = ["里","里面","里边","里头","中"]

    result = list()
    for _ in range(data_num):
        num = random.randint(1,100)
        template = []
        # 无槽语句 - 3%
        if num <= 3:
            template.append(add_T(["就"]+[""]*2,["随便","随意","任意","随机"],["给我","给","帮我"]+[""]*2,num_piece,c_end+[""]*8,["吧"]+[""]*2))
            template.append(add_T(["随便","随意","任意","随机"]+[""]*4,["都行", "都可以", "无所谓"],["的",""]))
            template.append(add_T(q_piece+[""]*2,["其他","别的","其他的","另外的","更多","更多的"],["专辑","歌手"]+c_end+[""]*8,["吗","没有","没"]+[""]*3))
            template.append(add_T(["还有","有"],["些什么","什么","哪些"], ["专辑", "歌手"]+c_end+[""]*8))
            template.append(add_T(c_start,["一首","几首","首","个"]+[""]*4,c_end+[""]*8))
            template.append(add_T(c_start+[""]*30, ["一首", "几首", "首", "个"]+[""]*4, c_end))
        # 状态转换匹配-事物 - 8%
        elif num <= 11:
            template.append(add_T(["那就", "那", "就"],["<actor>", "<song>"]*2+["<album>"],["吧", "的吧", "的"]+[""]*3))
            template.append(add_T(["那就", "那", "就"]+[""]*3, ["<actor>", "<song>"]*2+["<album>"],["吧", "的吧", "的"]))
            template.append(add_T(["那就", "那", "就"] + [""]*2,["给我","给","帮我"] + [""]*2,num_piece+search,["<actor>", "<song>"]*2+["<album>"],
                      ["吧", "的吧", "的"] + [""]*2))
        # 状态转换匹配-标签 - 4%
        elif num <= 15:
            template.append(add_T(["那就", "那", "就"],before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的",c_end+[""]*8,["吧",""]))
            template.append(add_T(["那就", "那", "就"],before_label_3+[""]*2, ["<label3>"], after_label_3+[""] * 3,
                                  c_end+[""]*8, ["吧", ""]))
            template.append(add_T(["那就", "那", "就"]+[""]*3,["<label2>"],["吧", "的吧", "的"]))
            template.append(add_T(["那就", "那", "就"],["<label2>"],["吧", "的吧", "的"]+[""]*3))
            template.append(add_T(["那就", "那", "就"]+[""]*3,["<label4>"],after_label_4,["吧", "的吧", "的"]))
            template.append(add_T(["那就", "那", "就"],["<label4>"],after_label_4,["吧", "的吧", "的"]+[""]*3))
        # 简单逻辑匹配-事物 - 8%
        elif num <= 23:
            template.append(add_T(q_piece,other+[""]*4,["<actor>", "<album>"],["的呢","的吗","的没"]))
            template.append(add_T(c_start+[""]*30,other, ["<actor>", "<album>"], ["的"],c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,other, ["<actor>", "<album>"], ["的",""],q_end+[""]*4))
        # 简单逻辑匹配-标签 - 4%
        elif num <= 27:
            template.append(add_T(q_piece,other+[""]*4,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的",c_end+[""]*8,["呢","吗",""]))
            template.append(add_T(c_start+[""]*30,other,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的",c_end+[""]*8))
            template.append(add_T(q_start + [""] * 8, q_piece + [""] * 4, other, before_label_1 + [""] * 10, ["<label1>"],
                                  after_label_1 + [""] * 3,"的", c_end + [""] * 8,q_end + [""] * 4))
            template.append(add_T(q_piece,other+[""]*4,["<label4>"],after_label_4,["呢", "吗", ""]))
            template.append(add_T(c_start+[""]*30,other,["<label4>"],after_label_4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,other,["<label4>"],after_label_4,q_end+[""]*4))
        # 指代匹配-人物 - 5%
        elif num <= 32:
            it_base = ["他", "她", "它", "别人", "其他人"]
            it = list()
            it.extend([i + "和<actor>" for i in it_base])
            it.extend([i + "跟<actor>" for i in it_base])
            it.extend(["<actor>跟" + i for i in it_base])
            it.extend(["<actor>和" + i for i in it_base])
            it.extend(["别人和" + i for i in ["他", "她", "它"]])
            it.extend([i + "和其他人" for i in ["他", "她", "它"]])
            it.extend(it_base*6)
            template.append(add_T(c_start+[""]*30, it, a_m_song,c_end+[""]*8))
            template.append(add_T(c_start+[""]*30, it, a_m_song,["<label4>"],after_label_4))
            template.append(add_T(c_start+[""]*30, it, a_m_song, ["<song>", "<album>"]))
            template.append(add_T(q_piece,it,a_m_song,c_end+[""]*8,["呢","吗","没"]+[""]*2))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,it,a_m_song,c_end+[""]*8,["呢","吗","没"]+[""]*2))
            template.append(add_T(q_piece,it,a_m_song,["<song>", "<album>"],["呢","吗","没"]+[""]*2))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,it,a_m_song,["<song>", "<album>"],["呢","吗","没"]+[""]*2))
        # 指代匹配-物品 - 5%
        elif num <= 37:
            template.append(add_T(c_start+[""]*30, before_album+[""]*4,"专辑",album_piece, ["的"], c_end+[""]*8))
            template.append(add_T(c_start+[""]*30, before_album+[""]*4,"专辑",album_piece, ["的"], ["<song>"]))
            template.append(add_T(c_start+[""]*30, before_album+[""]*4,"专辑",album_piece, ["的"], ["<label4>"],after_label_4))
            template.append(add_T(q_start+[""]*8,before_album+[""]*4,"专辑",album_piece, ["的"], c_end+[""]*8,q_end))
            template.append(add_T(q_start+[""]*8,before_album+[""]*4,"专辑",album_piece, ["的"], ["<song>"],q_end))
            template.append(add_T(q_start+[""]*8,before_album+[""]*4,"专辑",album_piece, ["的"], ["<label4>"],after_label_4,q_end))
        # 标签 - 8%
        elif num <= 45:
            template.append(add_T(c_start+num_piece+[""]*30, ["<label1>"], ["的"], ["<label2>"], ["的",""], c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,q_piece,before_label_1+[""]*10,["<label1>"],["的"], ["<label2>"], ["的",""], c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,before_label_1+[""]*10,["<label1>"],["的"], ["<label2>"], ["的",""], c_end+[""]*8,q_end+[""]*4))
            template.append(add_T(c_start+num_piece+[""]*30,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的", c_end+[""]*8))
            template.append(add_T(c_start+num_piece+[""]*30,["<label2>"],["吧", "的吧", "的"]+[""]*3))
            template.append(add_T(c_start+num_piece+[""]*30,before_label_3+[""]*2,["<label3>"],after_label_3+[""]*3,c_end+[""]*8))
            template.append(add_T(c_start+num_piece+[""]*30,["<label4>"],after_label_4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的",c_end+[""]*8,q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,["<label2>"],q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,before_label_3+[""]*2,["<label3>"],after_label_3+[""]*3,c_end+[""]*8,q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,["<label4>"],after_label_4,q_end+[""]*4))
            template.append(add_T(c_start+num_piece+[""]*30, a_actor,a_m_song,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的", c_end+[""]*8))
            template.append(add_T(c_start+num_piece+[""]*30, ["<album>"],["里","里面","里边","里头","中","中的"]+["的"]*4,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3, c_end+[""]*8))
            template.append(add_T(c_start+num_piece+[""]*30, a_actor,a_m_song,["<label2>"], ["吧", "的吧", "的"]+[""]*3))
            template.append(add_T(c_start+num_piece+[""]*30, ["<album>"],["里","里面","里边","里头","中","中的"]+["的"]*4,["<label2>"], ["吧", "的吧", "的"]+[""]*3))
            template.append(add_T(c_start+num_piece+[""]*30, a_actor,a_m_song,["<label4>"],after_label_4))
            template.append(add_T(c_start+num_piece+[""]*30, ["<album>"],["里","里面","里边","里头","中","中的"]+["的"]*4,["<label4>"],after_label_4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,a_actor,a_m_song,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的", c_end+[""]*8,q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, ["<album>"],["里","里面","里边","里头","中","中的"]+["的"]*4,before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3, c_end+[""]*8,q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, a_actor,a_m_song,["<label2>"],q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, ["<album>"],["里","里面","里边","里头","中","中的"]+["的"]*4,["<label2>"],q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, a_actor,a_m_song,["<label4>"],after_label_4,q_end+[""]*4))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, ["<album>"],["里","里面","里边","里头","中","中的"]+["的"]*4,["<label4>"],after_label_4,q_end+[""]*4))
            template.append(add_T("<label4>",after_label_4))
            template.append(add_T("<label3>时可以听", ["哪些","什么","怎样的","什么样的","啥"], "歌曲"))
            template.append(add_T("<label3>", ["时", "的时候"], ["听啥", "听什么", "听哪些"], c_end,["适合","可以听","合适","好听","动听"]))
            template.append(add_T("<label3>时有什么",["适合","可以听","合适","好听","动听"],"的",c_end))
            template.append(add_T(c_start+num_piece+[""]*30, ["<label5>"],["的",""], c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,q_piece,["<label5>"],["的",""],c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,["<label5>"],["的",""],c_end+[""]*8,q_end))
            template.append(add_T(c_start+num_piece+[""]*30,["<label1>"],["的"]+[""]*2,c_end))
        # 影视作品 - 4%
        elif num <= 49:
            movie_piece = ["电影", "电视剧"]*3+["综艺", "歌剧", "音乐剧", "动漫", "偶像剧", "电视"]
            movie_end = ["主题曲", "片头曲", "片尾曲", "插曲", "配乐", "背景音乐","主题歌"]
            template.append(add_T(c_start+num_piece+[""]*30, movie_piece+[""]*8, "<movie>",["里的", "中的"]+["的"]*2+[""]*2,movie_end+[""]*3))
            template.append(add_T(c_start+num_piece+[""]*30, movie_piece, "<movie>",["里的", "中的"]+["的"]*2+[""]*2,c_end+[""]*8))
            template.append(add_T(movie_piece+[""]*8,"<movie>",["里的", "中的"]+["的"]*2+[""]*2,movie_end+[""]*3,q_end))
            template.append(add_T(q_start+[""]*8,q_piece,movie_piece+[""]*8,"<movie>",["里的", "中的"]+["的"]*2+[""]*2,movie_end+[""]*3,["吗","没",""]))
            template.append(add_T("<movie>",["里的", "中的"]+["的"]*2+[""]*4,movie_end))
            template.append(add_T("<movie>",movie_piece,["里的", "中的"]+["的"]*2+[""]*4,movie_end))
        # 专辑 - 5%
        elif num<= 54:
            template.append(add_T(c_start+num_piece+[""]*30,"<album>",before_album+[""]*4,"专辑",album_piece,"的",c_end+[""]*8))
            template.append(add_T(c_start+num_piece+[""]*30,"<album>",album_piece,"的",c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,q_piece,"<album>",before_album+[""]*4,"专辑",album_piece,"的",c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,"<album>",before_album+[""]*4,"专辑",album_piece,"的",c_end+[""]*8,q_end))
            template.append(add_T(q_start+[""]*8,q_piece,"<album>",album_piece,"的",c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,"<album>",album_piece,"的",c_end+[""]*8,q_end))
        # 歌曲 - 12%
        elif num <= 66:
            template.append(add_T(search+c_start+num_piece+[""]*30,"<song>",["吧"]+after_song+[""]*10))
            template.append(add_T(q_start+[""]*8,q_piece,"<song>",after_song+[""]*10))
            template.append(add_T(q_start+[""]*8,"<song>",after_song+[""]*10,q_end))
            template.append(add_T(search+c_start+num_piece+[""]*30,a_actor,a_m_song,"<song>",["吧"]+[""]*2))
            template.append(add_T(q_start+[""]*8,q_piece,a_actor,a_m_song,"<song>"))
            template.append(add_T(q_start+[""]*8,a_actor,a_m_song,"<song>",q_end))
        # 普通匹配-非常见 - 14%
        elif num <= 80:
            template.append(add_T(search, "<song>", ["的歌手","是谁唱的","谁唱的","有谁唱过","谁唱过"]))
            template.append(add_T(["不知道","不清楚","不确定","不一定","不明白","忘记","不肯定","不懂"], ["是否是", "是不是"], "<actor>", ["唱的", "演唱的"], c_end+[""]*8))
            template.append(add_T(["不知道","不清楚","不确定","不一定","不明白","忘记","不肯定","不懂"], ["是否是", "是不是"], ["<actor>", "<album>"],["的",""]))
            template.append(add_T(c_start+num_piece+[""]*30, "<actor>", ["作词", "作曲","编曲","作"],["的","的",""],c_end+[""]*8))
            template.append(add_T("<actor>", ["作词", "作曲","编曲","作"],["的","的",""],c_end+[""]*8,q_end))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4,"<actor>", ["作词", "作曲", "编曲", "作"], ["的", "的", ""], c_end + [""] * 8))
            template.append(add_T(search, ["<song>","<song>","<song>","这首歌"], ["","的"], ["作词","作曲","编曲"],["人","者",""],["是谁",""]))
            template.append(add_T("<actor>", ["作词的", "作曲的"], c_end+[""]*8, ["有吗", "有没有", "有哪些", "", "", ""]))
            template.append(add_T(c_start+num_piece+[""]*30, "<actor>", ["作词的", "作曲的"], ["<album>"]*3 + a_e_song + ["专辑"]))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, "<actor>", ["作词的", "作曲的"], ["<album>"]*3 + a_e_song + ["专辑"]))
        # 普通匹配-常见 - 20%
        else:
            template.append(add_T(search+c_start+num_piece+[""]*30, c_end+[""]*8, "<song>"))
            template.append(add_T(["给我","帮我","给"]+[""]*3,search,"<song>",["这一首","这首"],c_end+[""]*8))
            template.append(add_T(["给我","帮我","给"]+[""]*3,num_piece,c_end+[""]*8,["，<song>","<song>"]))
            template.append(add_T(c_start+num_piece+[""]*30, ["<album>", "<actor>", "<label1>"], "的", c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,q_piece,["<album>","<label1>"]+["<actor>"]*2, "的", c_end+[""]*8,["吗","呢",""]))
            template.append(add_T(q_start+[""]*8,["<album>","<label1>"]+["<actor>"]*2, "的", c_end+[""]*8,q_end))
            template.append(add_T(c_start+num_piece+[""]*30,a_actor,a_m_song, ["<album>", "<label1>"], "的",c_end+[""]*8))
            template.append(add_T(q_start+[""]*8,q_piece,a_actor,a_m_song, ["<album>", "<label1>"],"的",c_end+[""]*8,["吗","呢",""]))
            template.append(add_T(q_start+[""]*8,a_actor,a_m_song, ["<album>", "<label1>"],"的",c_end+[""]*8,q_end))
            template.append(add_T(c_start+num_piece+[""]*30,"<album>",before_album+[""]*4,["专辑"],album_piece,"的",a_e_song))
            template.append(add_T(["要不", "还是", "切换", "换回"], ["<actor>唱", "<song>", "<album>", "<label1>"], "的", ["吧","算了",""]))
            # 不稳定的句式,后置
            template.append(add_T(c_start+num_piece+[""]*30, ["<actor>", "<song>", "<album>", "<label1>"], ["的", ""], ["吧","好吗",""]))
            template.append(add_T(c_start+num_piece+[""]*30, ["<actor>", "<album>"], ["的", ""],
                                  before_label_1+[""]*10,["<label1>"],after_label_1+[""]*3,"的",c_end+["<label2>"],["吧","好吗",""]))
            template.append(add_T(c_start+num_piece+[""]*30, ["<actor>"]*3+["<album>"], ["的", ""],a_e_song))
            template.append(add_T(q_start+[""]*8,q_piece, ["<actor>"]*3+["<album>"], ["的", ""],a_e_song))
            template.append(add_T(q_start+[""]*8,["<actor>"]*3+["<album>"], ["的", ""],a_e_song,q_end))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, c_end, "<song>"))
            template.append(add_T(c_start+num_piece+[""]*30, c_end, "<song>",["吧","好吗"]+[""]*3))
            template.append(add_T(c_start+num_piece+[""]*30,"专辑","<album>",album_piece+[""]*10,["的歌", "的<song>", ""]))
            template.append(add_T(q_start, "<actor>", ["演唱的", "唱的", "唱过的", "一起唱的"], ["什么", "哪些"], c_end))
            template.append(add_T(q_start, ["<actor>", "<album>"], "的", c_end, "有", ["什么", "哪些"]))
            template.append(add_T(c_start+num_piece+[""]*30, "专辑<album>", "的", a_e_song))
            template.append(add_T(q_start, "<song>", ["所属的","属于哪张","是哪张","所属","在哪张","所在","的",""], "专辑"))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, "<actor>", ["的",""], "<label5>","专辑"))
            template.append(add_T(c_start+num_piece+[""]*30, "<actor>", ["的",""], "<label5>","专辑"))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, "<song>", ["","的"], "<label6>", ["","版本","版"]))
            template.append(add_T(c_start+num_piece+[""]*30, "<song>", ["", "的"], "<label6>", ["", "版本", "版"]))
            template.append(add_T(q_start+[""]*8,q_piece+[""]*4, "<label6>", ["","版本","版"], ["","的"], "<song>"))
            template.append(add_T(c_start+num_piece+[""]*30, "<label6>", ["","版本","版"], ["","的"], "<song>"))
        result.append(random.choice(template))

    # with codecs.open("data/template.txt", "w", encoding="utf-8") as f:
    data = []
    for sentence in result:
        if random.random()<0.2:
            actor1 = random.choice(actor_list_en).lower()+" "
        else:
            actor1 = random.choice(actor_list_ch)
        if random.random()<0.2:
            actor2 = random.choice(actor_list_en).lower()+" "
        else:
            actor2 = random.choice(actor_list_ch)
        if random.random()<0.2:
            song = random.choice(song_list_en).lower()+" "
        else:
            song = random.choice(song_list_ch)
        if random.random()<0.2:
            album = random.choice(album_list_en).lower()+" "
        else:
            album = random.choice(album_list_ch)
        movie = random.choice(movie_list)
        l1 = random.choice(label1)
        l2 = random.choice(label2)
        l3 = random.choice(label3)
        l4 = random.choice(label4)
        l5 = random.choice(label5)
        l6 = random.choice(label6)
        entity_list = []
        if sentence.find("<actor>") != -1:
            sentence = sentence.replace("<actor>", actor1, 1)
            entity_list.append((actor1.strip(), "Actor"))
        if sentence.find("<actor>") != -1:
            sentence = sentence.replace("<actor>", actor2, 1)
            entity_list.append((actor2.strip(), "Actor"))
        if sentence.find("<song>") != -1:
            sentence = sentence.replace("<song>", song, 1)
            entity_list.append((song.strip(), "Song"))
        if sentence.find("<movie>") != -1:
            sentence = sentence.replace("<movie>", movie, 1)
            entity_list.append((movie.strip(), "Movie"))
        if sentence.find("<label1>") != -1:
            sentence = sentence.replace("<label1>", l1, 1)
            entity_list.append((l1, "Label"))
        if sentence.find("<label2>") != -1:
            sentence = sentence.replace("<label2>", l2, 1)
            entity_list.append((l2, "Label"))
        if sentence.find("<label3>") != -1:
            sentence = sentence.replace("<label3>", l3, 1)
            entity_list.append((l3, "Label"))
        if sentence.find("<label4>") != -1:
            sentence = sentence.replace("<label4>", l4, 1)
            entity_list.append((l4, "Label"))
        if sentence.find("<label5>") != -1:
            sentence = sentence.replace("<label5>", l5, 1)
            entity_list.append((l5, "Label"))
        if sentence.find("<label6>") != -1:
            sentence = sentence.replace("<label6>", l6, 1)
            entity_list.append((l6, "Label"))
        if sentence.find("<album>") != -1:
            sentence = sentence.replace("<album>", album, 1)
            entity_list.append((album, "Album"))
        data.append(
            {
                "sentence": sentence,
                "entity": entity_list
            }
        )
            # f.write(sentence + "\n")
    if not istrain:
        with codecs.open("data/dev.json", 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
    return data

def add_T(*args):
    template = ""
    for arg in args:
        if type(arg) == list:
            template += random.choice(arg)
        else:
            template += arg
    return template

def dump_train_set():
    '''
    自动生成训练数据并导出
    '''
    data1 = read_ccks_train("data/record/ccks2018_train.txt")
    data2 = write_data(20000)
    with codecs.open("data/train.json", 'w', encoding='utf-8') as f:
        json.dump(data1+data2, f, indent=4, ensure_ascii=False)

def process_log():
    '''
    从日志中抽取NLU提取结果
    '''
    with open("data/record/default.log","r",encoding="UTF-8") as fr:
        lines = list()
        for line in fr.readlines():
            list_line = line.strip().split("\t")
            if list_line[3]=="nlu":
                lines.append(list_line[-1])

    write_list = list()
    index_list = list()
    for i,line in enumerate(lines):
        line = line.replace('\'','')
        list_line = line.split(",")
        list_line = list(map(lambda x:x[1:],list_line))
        def_slot = "{}"
        undef_slot = "{}"
        str_in = ""
        in_undef = False
        in_def = False
        for item in list_line:
            if in_undef:
                if not item.startswith("res_fuzz"):
                    undef_slot = undef_slot+","+item
                else:
                    in_undef = False
            if in_def:
                if not item.startswith("undef_slot"):
                    def_slot = def_slot+","+item
                else:
                    in_def = False
            if item.startswith("def_slot"):
                def_slot = item[item.index(":")+1:].strip()
                in_def = True
            elif item.startswith("undef_slot"):
                undef_slot = item[item.index(":") + 1:].strip()
                in_undef = True
            elif item.startswith("str_in"):
                str_in = item[item.index(":") + 1:].strip()
        write_str = "{}\tdef_slot:{}\tundef_slot:{}".format(str_in,def_slot,undef_slot)
        if write_str not in write_list:
            write_list.append(write_str)
            index_list.append(i+1)
    final_list = list()
    for i,line in zip(index_list,write_list):
        final_list.append(str(i)+"\t"+line)

    with open("data/record/testCase_online.txt", "w", encoding="UTF-8") as fw:
        fw.write("\n".join(final_list))

def dump_test_set():
    fr = open("data/record/testCase_test.txt", "r", encoding="UTF-8")
    data = list()
    for line in fr.readlines():
        text,slot = line.strip().split("\t")
        slot_list = list()
        split_slot = slot[1:-1].split(",")
        for s in split_slot:
            if s:
                offset = s.find(":")
                type,value = s[:offset],s[offset+1:]
                if s=="":
                    print("slot:{} value:{}".format(type,value))
                value = value.strip()
                slot_list.append((value,type))
        data.append(
            {
                "sentence": text,
                "entity": slot_list
            }
        )
    fr.close()

    with codecs.open("data/test.json", 'w', encoding='utf-8') as fw:
        json.dump(data, fw, indent=4, ensure_ascii=False)


if __name__=="__main__":
    # write_segment_dict()
    dump_train_set()