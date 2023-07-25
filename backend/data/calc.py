import os
import time
import torch
import json
import logging

from geomloss import SamplesLoss
from multiprocessing import Process, Queue


class CalcDist:
    def __init__(self):
        self.sinkhorn = SamplesLoss('sinkhorn', blur=0.05, p=1)
        self.device = 'cpu'
        self.number_of_processes = 8
        self.problems, self.problem_index = self.get_problems()
        self.modes = ['DDMOP1', 'DDMOP3', 'DDMOP4', 'DDMOP5', 'DDMOP6', 'DTLZ1', 'DTLZ4', 'DTLZ5', 'DTLZ6']

    def __call__(self):
        TASK = self.get_tasks()
        task_queue = Queue()
        done_queue = Queue()

        for task in TASK:
            task_queue.put(task)

        for task in TASK:
            print(task)

        for i in range(self.number_of_processes):
            p = Process(target=self.worker, args=(task_queue, done_queue))
            p.daemon = True
            p.start()

        for i in range(len(TASK)):
            ret = done_queue.get()

        for i in range(self.number_of_processes):
            task_queue.put('STOP')
        return

    def get_problems(self):
        _, dirs, _ = os.walk('.').__next__()
        d = dict()
        for _dir in dirs:
            if _dir == 'zip':
                continue
            with open(f'../data/{_dir}/index.json') as _config:
                config = dict(json.load(_config))
                d[_dir] = config
            try:
                os.mkdir(f'../data/{_dir}/distance')
            except FileExistsError:
                pass
        return dirs, d

    def get_tasks(self):
        task_list = list()
        for mode in self.modes:
            if mode in self.problems:
                for index_1, value_1 in enumerate(self.problem_index[mode]['algorithms']):
                    for index_2, value_2 in enumerate(self.problem_index[mode]['algorithms']):
                        if index_1 > index_2:
                            continue
                        target_path = f'./{mode}/distance/{value_1}_{value_2}_{mode}_distance.json'
                        if os.path.isfile(target_path):
                            # print(target_path)
                            continue
                        task_list.append((self.dist, (mode, value_1, value_2)))
        return task_list

    def worker(self, input: Queue, output: Queue):
        for func, args in iter(input.get, 'STOP'):
            result = self.calculate(func, args[0], args[1], args[2])
            output.put([result, args[0], args[1], args[2]])

    def calculate(self, func, problem: str, algorithm_1: str, algorithm_2: str):
        return func(problem, algorithm_1, algorithm_2)

    def dist(self, problem: str, algorithm_1: str, algorithm_2: str):
        start_time = time.time()
        print(f'calculate start for algorithms {algorithm_1} and {algorithm_2} with problem {problem}')
        data_1, data_2, dist1to2, dist2to1 = self.read_file(problem, algorithm_1, algorithm_2)
        if algorithm_1 == algorithm_2:
            record = set()
            for index_1, value_1 in data_1.items():
                for index_2, value_2 in data_2.items():
                    if (index_1, index_2) in record:
                        dist1to2[index_1][index_2] = dist1to2[index_2][index_1]
                        dist2to1[index_2][index_1] = dist2to1[index_1][index_2]
                    # try:
                    try:
                        result = self.sinkhorn(value_1, value_2)
                        dist1to2[index_1][index_2] = result.item()
                        dist2to1[index_2][index_1] = result.item()
                        record.add((index_2, index_1))
                    except Exception:
                        print(f"EXCEPTION-{problem}-{algorithm_1}-{algorithm_2}")
                    # except ValueError:
                    #     print(value_1)
                    #     print(value_2)
                    #     print(len(value_1))
                    #     print(len(value_2))
            with open(f'./{problem}/distance/{algorithm_1}_{algorithm_2}_{problem}_distance.json', 'w') as target_file:
                json.dump(dist1to2, target_file)
        else:
            for index_1, value_1 in data_1.items():
                for index_2, value_2 in data_2.items():
                    try:
                        result = self.sinkhorn(value_1, value_2)
                        dist1to2[index_1][index_2] = result.item()
                        dist2to1[index_2][index_1] = result.item()
                    except Exception:
                        print(f"EXCEPTION-{problem}-{algorithm_1}-{algorithm_2}")
            with open(f'./{problem}/distance/{algorithm_1}_{algorithm_2}_{problem}_distance.json',
                      'w') as target_file_1:
                json.dump(dist1to2, target_file_1)
            with open(f'./{problem}/distance/{algorithm_2}_{algorithm_1}_{problem}_distance.json',
                      'w') as target_file_2:
                json.dump(dist2to1, target_file_2)
        end_time = time.time()
        elapse = end_time - start_time
        print(
            f'calculate finish for algorithms {algorithm_1} and {algorithm_2} with problem {problem} with elapse time {elapse}')
        return True

    def read_file(self, problem: str, algorithm_1: str, algorithm_2: str):
        path_1 = self.problem_index[problem]['algorithms'][algorithm_1]
        path_2 = self.problem_index[problem]['algorithms'][algorithm_2]
        with open(f'./{problem}/origin/{path_1}') as f:
            data_1 = dict(json.load(f))['result']['obj']
        with open(f'./{problem}/origin/{path_2}') as f:
            data_2 = dict(json.load(f))['result']['obj']
        dist1to2 = {k: {} for k in data_1.keys()}
        dist2to1 = {k: {} for k in data_2.keys()}
        return {k: torch.tensor(v, dtype=torch.float64) for k, v in data_1.items()}, \
            {k: torch.tensor(v, dtype=torch.float64) for k, v, in data_2.items()}, \
            dist1to2, dist2to1


class CalcIndividualDist:
    def __init__(self, *args):
        self.problem = args[0]
        self.algorithm_1 = args[1]
        self.algorithm_2 = args[2]
        self.sinkhorn = SamplesLoss('sinkhorn', blur=0.05, p=1)
        self.device = 'cpu'
        self.DATA_1, self.DATA_2 = self.read_file()
        self.len_1 = len(self.DATA_1)
        self.len_2 = len(self.DATA_2)
        self.total_len = self.len_1 + self.len_2
        self.number_of_processes = os.cpu_count()
        self.logger = logging.getLogger(__name__)

    def read_file(self):
        for cur, dirs, files in os.walk(f'./{self.problem}/origin/'):
            for file in files:
                if self.algorithm_1 == file.split('_')[0]:
                    with open(f'./{self.problem}/origin/{file}') as f:
                        self.logger.log(level=2, msg=f'get data from file: {file}')
                        DATA_1 = dict(json.load(f)['obj'])
                        DATA_1.pop('0')
                elif self.algorithm_2 == file.split('_')[0]:
                    with open(f'./{self.problem}/origin/{file}') as f:
                        self.logger.log(level=2, msg=f'get data from file: {file}')
                        DATA_2 = dict(json.load(f)['obj'])
                        DATA_2.pop('0')
        return torch.tensor(list(DATA_1.values())), torch.tensor(list(DATA_2.values()))

    def __call__(self):
        dist = torch.zeros(self.total_len, self.total_len, device=self.device)
        TASK1 = [(self.sinkhorn, (i, j, i, self.len_1 + j)) \
                 for i in range(self.len_1) for j in range(self.len_2)]
        TASK2 = [(self.sinkhorn, (i, j, i, j)) for i in range(self.len_1) \
                 for j in range(i, self.len_1)]
        TASK3 = [(self.sinkhorn, (i, j, self.len_1 + i, self.len_2 + j)) \
                 for i in range(self.len_2) for j in range(i, self.len_2)]

        task_queue = Queue()
        done_queue = Queue()

        for task in TASK1:
            task_queue.put(task)
        for i in range(self.number_of_processes):
            p = Process(target=self.worker, args=(task_queue, done_queue))
            p.daemon = True
            p.start()
        for i in range(len(TASK1)):
            ret = done_queue.get()
            dist[ret[1], ret[2]] = ret[0]
            dist[ret[2], ret[1]] = ret[0]

        for task in TASK2:
            task_queue.put(task)
        for i in range(self.number_of_processes):
            p = Process(target=self.worker, args=(task_queue, done_queue))
            p.daemon = True
            p.start()
        for i in range(len(TASK2)):
            ret = done_queue.get()
            dist[ret[1], ret[2]] = ret[0]
            dist[ret[2], ret[1]] = ret[0]

        for task in TASK3:
            task_queue.put(task)
        for i in range(self.number_of_processes):
            p = Process(target=self.worker, args=(task_queue, done_queue))
            p.daemon = True
            p.start()
        for i in range(len(TASK3)):
            ret = done_queue.get()
            dist[ret[1], ret[2]] = ret[0]
            dist[ret[2], ret[1]] = ret[0]

        for i in range(self.number_of_processes):
            task_queue.put('STOP')
        return dist

    def worker(self, input: Queue, output: Queue):
        for func, args in iter(input.get, 'STOP'):
            result = self.calculate(func, args[0], args[1])
            output.put([result, args[2], args[3]])

    def calculate(self, func, i, j):
        return func(self.DATA_1[i], self.DATA_2[j])


if __name__ == '__main__':
    calc = CalcDist()
    calc()
