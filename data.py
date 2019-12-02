# -*- coding:utf-8 -*-
"""
Created on 2019-11-15 09:37:58
Author: Xiong Zecheng (295322781@qq.com)
"""
import json
import random
import codecs

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
    fengge = ["纯音乐","摇滚","古典","民谣","舞曲","民歌","爵士","电子","R&B","说唱","戏曲","古风","乡村","蓝调","嘻哈","轻音乐","拉丁","雷鬼","新世纪","歌剧","中国风","男声","女声"]
    changjing = ["校园","雨天","开车","旅行","运动","咖啡厅","毕业","夜店","工作","睡前","派对","熬夜","学习","熬夜","午后","购物","坐车"]
    qinggan = ["伤感","快乐","思念","寂寞","失恋","甜蜜","祝福","感动","抒情","恋爱","安静","温暖","激情","怀念","虐心","轻松","愤怒","怀旧","悠扬","搞笑","宣泄","情侣"]
    zhuti = ["铃声","经典","分手","军旅","友情","红歌","KTV","亲情","儿歌","成名曲","背景","佛教","武侠","网络","胎教","方言"]
    yueqi = ["钢琴","吉他","古筝","小提琴","琵琶","二胡","萨克斯","葫芦丝","古琴"]
    tezhi = ["青春","故乡","励志","草原","治愈","文艺","清新","异域","神曲","雷人","独自"]
    yuyan = ["国语","华语","粤语","英语","英文","闽南语","俄语","外语","中文","潮汕话","泰国","港台","普通话"]
    niandai = [str(i) + "0年代" for i in range(0,10)]
    # 形容词类型标签
    label1 = ["摇滚","古典","电子","R&B"] + qinggan + tezhi + changjing + niandai

    # 名词类型标签
    label2 = fengge + zhuti + yueqi + yuyan

    # 年代+语言，可以修饰专辑的标签
    label3 = niandai + yuyan

    # 可以修饰音乐的标签
    label4 = yueqi + ["纯音乐","现场","轻音乐"] + yuyan

    # 句子正常开头，有""字段是因为直接询问"周杰伦的稻香"这种情况出现是没有开头的
    q_start = ["请问有", "请给我播", "请给我来一首", "给我来一首", "来一首", "播放", "播", "有", "有没有", "请播一首", "我想听",
               "想听", "能来一首", "能播", "可以播", "可以来一首", "再来一首", "给我播放", "给我来一首", "给我播", "能不能播", "搜索"
               "再播", "推荐几首", "查", "查询", "找一首", "给我播放", "放", "来首", "来一个", "找", "那", "那就播放", "那放", "搜"
               "放一首", "找一下", "切成", "切换成", "其它", "别的", "那就", "我要", "要", "帮我找", "帮我查一下", "帮我播一首",
               "", "", "", "", "", "", "", "", "", ""]
    search = ["请问", "搜索", "查", "查询", "找一首", "找", "搜", "我要", "帮我找", "帮我查一下", "", "", "", "", "", "", "", "", "", ""]
    before_label = ["适合", "比较", "有点", "稍微", "非常", "很", "", "", "", "", ""]
    after_label_1 = ["的时候听的","时听的","的时候放的","时放的","听的","放的","一点的","风格的","的","的","的","的"]
    after_label_2 = ["音乐","歌曲","歌","曲","乐曲","曲子","配乐"]

    # 问歌手，重复<actor>，是因为一般问一个歌手的频率比较高，让问合唱的与一个的频率相同
    a_actor = ["<actor>和<actor>", "<actor>或者<actor>", "<actor>、<actor>", "<actor>和<actor>", "<actor>与<actor>",
               "<actor>，<actor>", "<actor>跟<actor>","<actor>", "<actor>", "<actor>", "<actor>", "<actor>", "<actor>"]
    # 歌手和歌的搭配
    a_m_song = ["合唱的", "演唱的", "唱的", "唱过的", "一起唱的", "版本的", "演唱会的", "音乐会的", "的", "的", "的", "的", "的"]
    a_e_song = ["<song>", "<song>", "<song>", "歌", "", "歌曲", "音乐"]
    modal = ["吧", "吗", "呢", "有吗", "没有吗", "有没有", "", "", ""]
    c_end = ["音乐", "歌曲", "曲子", "歌", "乐曲", "配乐", "", "", ""]
    result = []
    for _ in range(data_num):
        num = random.randint(0,20)
        template = []
        # 问歌曲 - 4
        if num <= 3:
            template.append(add_T(q_start, ["歌手","歌手是","","","","","","",""], a_actor, a_m_song, a_e_song, modal))
        # 状态转换匹配 - 2
        elif num <= 5:
            template.append(add_T(["那就","那","就","",""], ["<actor>", "<song>"], ["吧","的吧","的","",""]))
            template.append(add_T(["还有",""],["别的","其它的","",""],["<actor>", "<album>", "<song>"],["呢","吗","的呢",""]))
        # 指代匹配 - 3
        elif num <= 8:
            it_base = ["他", "她", "它", "别人", "其他人"]
            it = list()
            it.extend([i + "和<actor>" for i in it_base])
            it.extend([i + "跟<actor>" for i in it_base])
            it.extend(["<actor>跟" + i for i in it_base])
            it.extend(["<actor>和" + i for i in it_base])
            it.extend(["别人和" + i for i in ["他", "她", "它"]])
            it.extend([i + "和其他人" for i in ["他", "她", "它"]])
            it.extend(it_base)
            template.append(add_T(q_start, it, a_m_song, ["歌", "", "歌曲"]))
            template.append(add_T(q_start, it, a_m_song, "<label1>", after_label_1, after_label_2))
            template.append(add_T(q_start, it, a_m_song, ["<song>", "<album>"]))
            template.append(add_T(q_start, ["这", "那", "此", ""], ["专辑", "专辑中", "专辑里"], "的", a_e_song))
            template.append(add_T(q_start, ["这", "那", "此", ""], ["专辑", "专辑中", "专辑里"], "的", "<label1>", after_label_1, after_label_2))
        # 标签 - 3
        elif num <= 11:
            template.append(add_T(q_start, "<label1>", "的", "<label2>", ["的", "的", ""], after_label_2))
            template.append(add_T(q_start, "<label>", after_label_1))
            template.append(add_T(q_start, ["<actor>", "<album>"], ["演唱的", "唱的", "的"], "<label>", after_label_2))
        # 普通匹配 - 9
        else:
            template.append(add_T(search, "<song>", ["的歌手","是谁唱的","谁唱的","有谁唱过","谁唱过"]))
            template.append(add_T(["不知道", "不清楚", "不确定", "不一定"], ["是", "是不是"], "<actor>", ["唱的", "演唱的"], c_end))
            template.append(add_T(["不知道", "不清楚", "不确定", "不一定"], ["是", "是不是"], ["<actor>", "<album>"]))
            template.append(add_T(q_start, c_end, "<song>"))
            template.append(add_T(["查", "找", "查找"], "一下<song>", "这首歌"))
            template.append(add_T(q_start, "<actor>", ["作词", "作曲","编曲","作"],["的","的",""], c_end))
            template.append(add_T(search, ["<song>","<song>","<song>","这首歌"], ["","的"], ["作词","作曲","编曲"],["人","者",""],["是谁","是",""]))
            template.append(add_T("<actor>", ["作词的", "作曲的"], c_end, ["有吗", "有没有", "有哪些", "", "", ""]))
            template.append(add_T(q_start, "<actor>", ["作词的", "作曲的"], ["<album>" for i in range(3)] + a_e_song + ["专辑"]))
            template.append(add_T(q_start, ["电影", "电视剧", "综艺", "歌剧", "音乐剧", "动漫", "偶像剧", "电视", "", "", ""], "<movie>", ["", "的"],["主题曲", "片头曲", "片尾曲", "插曲", "配乐", "背景音乐"]))
            template.append(add_T(q_start, "<movie>", ["", "的"], ["电影", "电视剧", "综艺", "歌剧", "音乐剧", "动漫", "偶像剧", "电视", "", "", ""], ["", "的"],["主题曲", "片头曲", "片尾曲", "插曲", "配乐", "背景音乐"]))
            template.append(add_T(q_start, ["适合", "适合我", "让人", "充满", "", "", ""], "<label1>", ["的时候", "时", "一点", "", "", ""], "听的", c_end))
            template.append(add_T(q_start, ["适合", "适合我", "让人", "充满", "", "", ""], "<label1>", after_label_1, after_label_2))
            template.append(add_T(q_start, ["比较", "有点", "稍微", "有关", "非常", "很"], "<label1>的", c_end))
            template.append(add_T(q_start, ["比较", "有点", "稍微", "有关", "非常", "很", "", "", "", ""], "<label1>", after_label_1, after_label_2))
            template.append(add_T(q_start, a_actor, a_m_song, "<label1>", after_label_1, after_label_2))
            template.append(add_T(q_start, "<album>", ["这张", "那张", "这个", "那个", "", ""], ["专辑", "专辑中", "专辑里"], "的<label1>", after_label_1, after_label_2))
            template.append(add_T(q_start, "<album>", ["里面", "中", "里"], "的<label1>", after_label_1, after_label_2))
            template.append(add_T(q_start, before_label, "<label1>", after_label_1, "<label2>", c_end))
            template.append(add_T("<label1>", after_label_1, after_label_2, ["有没有", "有吗", ""]))
            template.append(add_T("<label>歌曲"))
            template.append(add_T("<label1>时可以听", ["哪些", "什么"], "歌曲"))
            template.append(add_T("<label1>", ["时", "的时候"], ["听啥", "听什么", "听哪些"], "歌", ["好", "", "合适"]))
            template.append(add_T("<label1>时有什么合适的歌"))
            template.append(add_T(q_start, "<album>", ["这张", "那张", "这个", "那个", "", ""], ["专辑", "专辑中", "专辑里"], c_end))
            template.append(add_T(q_start, "<album>", ["里面", "中", "里", ""], c_end))
            template.append(add_T(q_start, ["<album>", "<actor>", "<label>"], "的", c_end))
            template.append(add_T(q_start, "<actor>", ["演唱的", "唱的", "的"], ["<album>", "<label1>"], "的", c_end))
            template.append(add_T(q_start, "<album>", ["这张", "那张", "这个", "那个", "", ""], ["专辑", "专辑中", "专辑里"], a_e_song))
            template.append(add_T(["要不", "还是", "切换", "换回"], ["<actor>唱", "<song>", "<album>", "<label1>"], "的", modal))
            # 不稳定的句式,后置
            template.append(add_T(q_start, ["<actor>", "<song>", "<album>", "<label1>"], ["的", ""], modal))
            template.append(add_T(q_start, ["<actor>", "<album>"], ["的", ""], "<label1>", after_label_1 + ["" for _ in range(5)], after_label_2 + ["" for _ in range(5)]))
            template.append(add_T(q_start, ["<actor>", "<album>"], ["的", ""], "<song>"))
            template.append(add_T(q_start, after_label_2, "<song>"))
            template.append(add_T(q_start, "专辑", "<album>", ["的歌", "的<song>", ""]))
            template.append(add_T(["请问", ""], "<actor>", ["演唱的", "唱的", "唱过的", "一起唱的"], ["什么", "哪些"], c_end))
            template.append(add_T(["请问", ""], ["<actor>", "<album>"], "的", c_end, "有", ["什么", "哪些"]))
            template.append(add_T(q_start, "专辑<album>", "的", a_e_song))
            template.append(add_T(q_start, "<song>", ["所属的","属于哪张","是哪张","所属","在哪张","所在","的",""], "专辑"))
            template.append(add_T(q_start, "<actor>", ["的",""], "<label3>","专辑"))
            template.append(add_T(q_start, "<song>", ["","的"], "<label4>", ["","版本","版"]))
            template.append(add_T(q_start, "<label4>", ["","版本","版"], ["","的"], "<song>"))
        result.append(random.choice(template))

    # with codecs.open("data/template.txt", "w", encoding="utf-8") as f:
    data = []
    for sentence in result:
        if random.randint(0, 1):
            actor1 = random.choice(actor_list_en).lower()
        else:
            actor1 = random.choice(actor_list_ch)
        if random.randint(0, 1):
            actor2 = random.choice(actor_list_en).lower()
        else:
            actor2 = random.choice(actor_list_ch)
        if random.randint(0, 1):
            song = random.choice(song_list_en).lower()
        else:
            song = random.choice(song_list_ch)
        if random.randint(0, 1):
            album = random.choice(album_list_en).lower()
        else:
            album = random.choice(album_list_ch)
        movie = random.choice(movie_list)
        l1 = random.choice(label1)
        l2 = random.choice(label2)
        l3 = random.choice(label3)
        l4 = random.choice(label4)
        l = random.choice(label1 + label2)
        entity_list = []
        if sentence.find("<actor>") != -1:
            sentence = sentence.replace("<actor>", actor1, 1)
            entity_list.append((actor1, "Actor"))
        if sentence.find("<actor>") != -1:
            sentence = sentence.replace("<actor>", actor2, 1)
            entity_list.append((actor2, "Actor"))
        if sentence.find("<song>") != -1:
            sentence = sentence.replace("<song>", song, 1)
            entity_list.append((song, "Song"))
        if sentence.find("<movie>") != -1:
            sentence = sentence.replace("<movie>", movie, 1)
            entity_list.append((movie, "Movie"))
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
        if sentence.find("<label>") != -1:
            sentence = sentence.replace("<label>", l, 1)
            entity_list.append((l, "Label"))
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
    data1 = read_ccks_train("data/ccks2018_train.txt")
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
    data1 = read_ccks_train("data/record/ccks2018_train.txt")
    print(len(data1))