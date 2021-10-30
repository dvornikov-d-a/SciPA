class Validation:
    def __init__(self, model, df, train_dataset_volumes, shuffle_count=5):
        self.model = model
        self.original_df = df
        # Должны быть кратны максимальному, иначе не все объекты попадут в обучающую выборку (исправить)
        self.train_dataset_volumes = train_dataset_volumes
        self.test_dataset_volume = len(df) - max(train_dataset_volumes)
        self.shuffle_count = shuffle_count
        self.res = {}
        for train_dataset_volume in self.train_dataset_volumes:
            self.res[train_dataset_volume] = {
                'min': {},
                'max': {},
                'mean': {},
                'median': {}
            }

    def validate(self):
        scores_all = {}
        for train_dataset_volume in self.train_dataset_volumes:
            scores_all[train_dataset_volume] = []
        for i in range(self.shuffle_count):
            current_df = self.original_df.sample(frac=1).reset_index(drop=True)
            current_test_dataset = current_df[-self.test_dataset_volume:]
            for train_dataset_volume in self.train_dataset_volumes:
                # Добавить возможность +1 перестановки, если не кратное число
                k = int((len(current_df) - len(current_test_dataset)) / train_dataset_volume)
                for j in range(k):
                    start_object = j * train_dataset_volume
                    end_object = start_object + train_dataset_volume
                    current_train_dataset = current_df[start_object:end_object]
                    self.model.fit(current_train_dataset)
                    self.model.test(current_test_dataset)
                    scores_all[train_dataset_volume].append(self.model.scores)
        for train_dataset_volume in self.train_dataset_volumes:
            scores = scores_all[train_dataset_volume]
            for metric in self.model.metrics:
                metric_values = [score[metric] for score in scores]
                self.res[train_dataset_volume]['min'][metric] = min(metric_values)
                self.res[train_dataset_volume]['max'][metric] = max(metric_values)
                self.res[train_dataset_volume]['mean'][metric] = numpy.mean(metric_values)
                self.res[train_dataset_volume]['median'][metric] = numpy.median(metric_values)