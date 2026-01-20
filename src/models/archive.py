class Archive:
    def __init__(self, np_archive):
        self.archive = np_archive

    def core_dataset(self):
        return self.archive.core_dataset
    
    def samples_for_each_song(self):
        return self.archive.samples_for_each_song
    
    def steps_per_duration(self):
        return self.archive.steps_per_duration[0]
    
    def freq_buckets(self):
        return self.archive.freq_buckets[0]