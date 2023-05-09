import re
import string

# These replacements are done on a per token basis
per_token_regexes = {
    re.compile("^(x+o+)+x*$"): "xoxo",
    re.compile("^(l+m+f*a+o+)+$"): "lmao",
    re.compile("^h+a+(h+a+)+h*$"): "haha",
    re.compile("^a+h+a+h+(a+h+)*a*$"): "haha",
    re.compile("^y+a+y+$"): "yay",
    re.compile("^(l+o+)+l+[zs]*$"): "lol",
    re.compile("^w+h*o+h+o+$"): "woohoo",
    re.compile("^a+ww+h*$"): "aww",
    re.compile("^(o+m+f*g+[ei]*)+$"): "omg",
    re.compile("^o+h*m+y*g+((o+d*)|(a+d*))$"): "omg",
    re.compile("^(p+l+[sz]+e*)*$"): "please",
    re.compile("^x+d+$"): "xd",
    re.compile("^t+h+x+$"): "thanks",
    re.compile("^i+d+g+a+s*f+$"): "idgaf",
    re.compile("^w+t+f+$"): "wtf",
    re.compile("^i+d+k+n?o?w?$"): "i do not know",
}

# These substitutions are applied on the entire tweet
global_regexes = {
    # re.compile("\\\\ [0-9][0-9][0-9]"): "",
        re.compile(r'(?<=\()\s*(\w)\s*(?=\))|\(\s*(\w)\s*\)|\(\s*(\w+)\s*\)'): r'\1\2\3',  # Removing spaces from within parentheses
        re.compile(r'(\d)[oO]'): r'\g<1>0',  # Transforming "1o" into "10", "2O" into "20", etc.
        re.compile(r'[oO]{3,}'): '000',  # Transforming "ooo" into "000", "OOO" into "000", etc.
        re.compile(r'(\d)\s*,\s*(\d)'): r'\1\2',  # Removing spaces between numbers and commas
        re.compile(r'(?<=\b\w)(\s+)(?=\w\b)'): '',  # Removing spaces between characters
        # re.compile(r'http?://\S+'): '',  # Removing URLs
        # re.compile(r'\brt\b'): '',  # Removing RT (retweet) indicator
        # re.compile(
        # r'[^\w\s\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF\U0001F1E0-\U0001F1FF]+'): ' ',
        # # Removing special characters except for emoticons
        re.compile(r'\s+'): ' ',  # Removing extra whitespace
    # re.compile(r'(\w)\1{2,}'): r'\1'  # Removing repeated letters
}

# global_regexes = {
#     re.compile(r'(?<=\()\s*(\w)\s*(?=\))|\((\w+)\)'): r'\1\2',  # Removing spaces from within parentheses and removing parentheses if they contain only one word or letter
#     re.compile(r'(\d)[oO]'): r'\g<1>0',  # Transforming "1o" into "10", "2O" into "20", etc.
#     re.compile(r'[oO]{3,}'): '000',  # Transforming "ooo" into "000", "OOO" into "000", etc.
#     re.compile(r'(\d)\s*,\s*(\d)'): r'\1\2',  # Removing spaces between numbers and commas
#     re.compile(r'([\?\!])\s*\1'): r'\1\1',  # Removing spaces between duplicate symbols such as "? ?" and "! !"
# }

# These are replaced on a per token basis
abbreviations = {
    "s / o": "shout-out",
    "pj's": "pyjamas",
    "bday": "birthday",
    "gr8": "great",
    "bc": "because",
    "dming": "direct messaging",
    "nxt": "next",
    "ppl": "people",
    "u": "you",
    "w /": "with",
    "w / o": "without",
    "w / e": "whatever",
    "convo": "conversation",
    "jk": "just kidding",
    "idc": "i do not care",
    "dw": "do not worry",
    "idk": "i do not know",
    "tbh": "to be honest",
    "irl": "in real life",
    "rn": "right now",
    "btw": "by the way",
}


# No idea what to do with
# - things like < < or << or >>>


# These are done for the entire tweet but
# `spaceout` is used to match whitespace within the patterns too
global_replacement_map = [
    (":/", "😕"),
    (":^)", "😊"),
    ("^_-", "😉"),
    ("O:-)", "😇"),
    (":v", "👍"),
    (":-*", "😗"),
    ("@_@", "😵"),
    ("B-)", "😎"),
    (":-/", "😕"),
    (":\\", "😕"),
    (":s", "😖"),
    (":[", "😞"),
    (":}", "😃"),
    (":{", "😞"),
    (":<", "😞"),
    (":>", "😃"),
    (";(", "😢"),

    ("=D", "😄"),
    ("=p", "😛"),
    ("=|", "😐"),
    (">:(", "😠"),
    (">:)", "😈"),
    (">:-(", "😠"),
    (">:D", "😆"),
    (">:[", "😠"),
    ("oO", "😲"),
    (";^)", "😉"),
    (":@", "😠"),
    # 😊
    ("^^", "😊"),
    ("^_^", "😊"),
    ("o^_^o", "😊"),
    ("^_^;", "😅"),
    ("^__^", "😊"),
    ("^___^", "😊"),
    ("^o^", "😊"),
    ("^-^", "😊"),
    ("^.^", "😊"),
    (":)", "😊"),
    ("):", "😊"),
    ("=)", "😃"),
    ("(=", "😊"),
    (":-)", "😊"),
    ("=]", "😊"),
    (":]", "😃"),
    (">:]", "😈"),
    (":-]", "😊"),
    ("=d", "😊"),  # =D
    (":D", "😄"),
    (":d", "😄"),
    (":-d", "😊"),  # :-D
    (":-D", "😄"),
    (";')", "😊"),
    ("[:", "😊"),
    # 🙁
    (":(", "🙁"),
    (":-(", "🙁"),
    (":c", "🙁"),
    (":-c", "🙁"),
    ("=(", "🙁"),
    ("=-(", "🙁"),
    ("=[", "🙁"),
    ("=-[", "🙁"),
    # 😙
    (":*", "😘"),
    (":x", "😙"),
    # 😑
    ("-.-", "😑"),  # Is this the correct interpretation?
    ("-_-", "😑"),
    ("'_'", "😑"),
    ("._.", "😑"),
    ("-___-", "😑"),
    ("*_*", "😑"),
    ("*___*", "😑"),
    ("-__-", "😑"),
    ("'.'", "😑"),
    ("'-'", "😑"),
    (":|", "😐"),
    (";/", "😑"),
    (":-|", "😑"),
    # 😉
    (">;)", "😈"),
    (";)", "😉"),
    (";-)", "😉"),
    (";]", "😉"),
    (";-]", "😉"),
    (";-d", "😉"),  # ;-D
    (";d", "😉"),  # ;D
    (";o)", "😉"),
    # 😢
    (":'(", "😢"),
    (":'-(", "😢"),
    (")':", "😢"),
    (":,(", "😢"),
    (";'(", "😢"),
    (";-;", "😢"),
    (";w;", "😢"),
    # 😂
    ("(':", "😂"),
    (":')", "😂"),
    (":'d", "😂"),  # :'D
    ("x-d", "😂"),
    (":']", "😂"),
    (";']", "😂"),
    # 😺
    (":'3", "😺"),
    (":3", "😺"),
    (":-3", "😺"),
    ("=3", "😺"),
    # 😖
    (">.<", "😖"),
    (">-<", "😖"),
    (">_<", "😣"),
    (">__<", "😖"),
    (">_>", "😏"),
    ("<_<", "😒"),
    (">__>", "😖"),
    ("<__<", "😖"),
    (">___>", "😖"),
    ("<___<", "😖"),
    (">->", "😖"),
    ("<-<", "😖"),
    (">.>", "😖"),
    ("<.<", "😖"),
    (":$", "🤑"),
    # 😲
    ("=-0", "😲"),
    ("=-o", "😲"),
    ("=0", "😲"),
    ("=o", "😲"),
    (":-0", "😲"),
    (":-o", "😲"),
    (":0", "😲"),
    (":o", "😮"),
    (":O", "😮"),
    (">:O", "😲"),
    # Other
    (":-?", "😛"),
    (";p", "😜"),
    (":P", "😛"),
    (":p", "😛"),
    ("8)", "😎"),
    ("8-)", "😎"),
    ("</3", "💔"),
    ("<3", "❤️"),
    # 😳
    ("o-o", "😳"),
    ("0-0", "😳"),
    ("o-0", "😳"),
    ("0-o", "😳"),
    ("o_o", "😳"),
    ("0_0", "😲"),
    ("O_O", "😲"),
    ("O_o", "😳"),
    ("o_O", "😳"),
    ("o_0", "😳"),
    ("0_o", "😳"),
    ("o.o", "😳"),
    ("0.0", "😳"),
    ("o.0", "😳"),
    ("0.o", "😳"),
    ("o.O", "😳"),
    # 💀
    ("x_x", "💀"),
    ("+_+", "💀"),
]

unique_emoticons = set()
for k, v in global_replacement_map:
    if " " in k:
        raise ValueError("emoticon", k, "has space")
    if k in unique_emoticons:
        raise ValueError("Duplicate emoticon", k)
    unique_emoticons.add(k)

# prefer longer matches
global_replacement_map.sort(key=lambda e: -len(e[0]))


# Uncensor curse words using static mapping
curse_word_uncensor = {
    " f * ckin ": " fuckin ",
    " c * nts ": " cunts ",
    " bullsh * t ": " bullshit ",
    " a * s ": " ass ",
    " p * ssing ": " pissing ",
    " motherf * cking ": " motherfucking ",
    " pus * y ": " pussy ",
    " b * tchez ": " bitchez ",
    " sh * tty ": " shitty ",
    " fuck * r ": " fucker ",
    " n * ggas ": " niggas ",
    " motherf * ckin ": " motherfuckin ",
    " n * gger ": " nigger ",
    " h * e ": " hoe ",
    " s * it ": " shit ",
    " a * sholes ": " assholes ",
    " ar * e ": " arse ",
    " k * ll ": " kill ",
    " d * ck ": " dick ",
    " b * tch ": " bitch ",
    " m * therf * cker ": " motherfucker ",
    " c * ck ": " cock ",
    " f * cks ": " fucks ",
    " fu * kin ": " fuckin ",
    " motherf * cker ": " motherfucker ",
    " b * llocks ": " bollocks ",
    " bit * hes ": " bitches ",
    " sh * g ": " shag ",
    " bit * h ": " bitch ",
    " s * x ": " sex ",
    " b * stard ": " bastard ",
    " f * cked ": " fucked ",
    " s * ck ": " suck ",
    " fuck * n ": " fuckin ",
    " da * n ": " damn ",
    " f * k ": " fuk ",
    " cu * t ": " cunt ",
    " jack * ss ": " jackass ",
    " f * ckers ": " fuckers ",
    " sh * ts ": " shits ",
    " b * tchy ": " bitchy ",
    " cr * p ": " crap ",
    " p * ssed ": " pissed ",
    " fu * ked ": " fucked ",
    " b * tching ": " bitching ",
    " b * stards ": " bastards ",
    " f * ck ": " fuck ",
    " p * ssy ": " pussy ",
    " bi * tch ": " biatch ",
    " tw * t ": " twat ",
    " n * gga ": " nigga ",
    " mothaf * ckin ": " mothafuckin ",
    " assh * le ": " asshole ",
    " s * ut ": " slut ",
    " w * nker ": " wanker ",
    " fu * k ": " fuck ",
    " f * cking ": " fucking ",
    " f * cker ": " fecker ",
    " nig * er ": " nigger ",
    " c * nt ": " cunt ",
    " fu * king ": " fucking ",
    " fuck * ng ": " fucking ",
    " sh * tter ": " shitter ",
    " p * rn ": " porn ",
    " bulls * it ": " bullshit ",
    " wa * ker ": " wanker ",
    " di * k ": " dick ",
    " f * ckstick ": " fuckstick ",
    " p * ssys ": " pussys ",
    " b * tches ": " bitches ",
    " f * ck * ng ": " fucking ",
    " sh * te ": " shite ",
    " u * i ": " uzi ",
    " h * ll ": " hell ",
    " sh * tt ": " shitt ",
    " sl * g ": " slag ",
    " b * lls ": " balls ",
    " sh * t ": " shit ",
    " d * mn ": " damn ",
    " p * ss ": " piss ",
    " bi * ch ": " bitc h",
}
