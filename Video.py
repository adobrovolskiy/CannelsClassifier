
class Video(object):
    def __init__(self, title, description, channel_id, category, tags):
        self.tags = tags
        self.category = category
        self.title = title
        self.description = description
        self.channelId = channel_id