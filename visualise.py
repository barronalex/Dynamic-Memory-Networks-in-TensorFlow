def print_story(self, story, length):
    for i in range(length):
        print self.ivocab[story[i]],
    print '--------------------------'


def visualize(self):
    for i in range(10):
        # first print the story
        self.print_story(self.train_input[i], self.train_input_lens[i])
