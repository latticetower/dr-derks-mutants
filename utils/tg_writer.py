"""Helper class to send data to telegram channel
"""
import os
import telegram


class TelegramPost(object):
    def __init__(self):
        self.text = []
        self.media = []
        self.tags = []

    def add_text(self, text):
        self.text.append(text)

    def add_param(self, key, value):
        self.text.append(f"{key}: {value}")

    def add_media(self, media):
        self.media.append(media)

    def set_tags(self, tags):
        self.tags = tags

    def send(self, bot, channel):
        text = "\n".join([str(x) for x in self.text])
        if len(self.media) > 0:
            with open(self.media[0], 'rb') as image:
                bot.send_photo(channel, photo=image, caption=text)
        else:
            bot.send_message(channel, text)


class PostWrapper(object):
    def __init__(self, bot, channel):
        self.post = None
        self.bot = bot
        self.channel = channel

    def __enter__(self):
        self.post = TelegramPost()
        return self.post

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is not None:
            return
        self.post.send(self.bot, self.channel)
        print(self.bot.get_me())


class TelegramWriter(object):
    def __init__(self, token, channel):
        self.token = token
        self.channel = channel
        self.bot = telegram.Bot(token=token)

    def post(self):
        return PostWrapper(self.bot, self.channel)


if __name__=="__main__":
    # sample run:
    token = os.environ.get('TELEGRAM_BOT_TOKEN', None)
    channel = os.environ.get('TELEGRAM_CHANNEL', None)
    writer = TelegramWriter(token, channel)
    with writer.post() as f:
        print(f)
        f.add_text("Smth wiked")
        f.add_param("smth", 0.4)
        f.add_media("example.png")