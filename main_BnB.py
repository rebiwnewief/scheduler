import sys
from itertools import combinations
from pathlib import Path

import pulp
import csv
import numpy as np
import pandas as pd


class TaskScheduler:
    def __init__(self, tasks):
        # self.tasks = sorted(tasks, key=lambda x: x['deadline'])
        self.tasks = tasks # .sort_values(by='deadline')
        self.n = len(tasks)
        self.best_solution = None
        self.best_reward = 0
        self.reward = self.tasks['reward']

    def read_tasks_from_csv(filename: str):
        tasks = []
        record = 0
        clear_filename = Path(filename).stem.split("_")
        if len(clear_filename) > 1:
            record = int(clear_filename[1])
        tasks = pd.read_csv(filename)
        return tasks, record
    
    def upper_bound_lp(self, selected_tasks):
        prob = pulp.LpProblem('Optimization_problem', pulp.LpMaximize)

        # максимальное время начала задачи
        # max_time = max([task['deadline'] for task in selected_tasks])
        max_time = selected_tasks['deadline'].max()

        # определим переменную: номер задания и время начала
        # в любой момент времени может начаться одна из N задач
        tasks = pulp.LpVariable.dicts(
            'task',
            ((task_id, t_start) for task_id in selected_tasks['id'] for t_start in range(max_time + 1)),
            cat='Binary'
        )
        #print(tasks)
        
        # каждая задача выполняется не более 1 раза
        for task_id in selected_tasks['id']:
            prob += pulp.lpSum(
                tasks[task_id, t_start] for t_start in range(max_time + 1)
            ) <= 1

        # в каждый момент времени выполняется не более 1 задачи
        for t_start in range(max_time + 1):
            prob += pulp.lpSum(
                tasks[task_id, t_start] for task_id in selected_tasks['id']
            ) <= 1
        
        # время старта + длительность <= дедлайна
        for _, task in selected_tasks.iterrows():
            for t_start in range(max_time + 1):
                # если время выполнения меньше дедлайна
                if task['duration'] <= task['deadline']:
                    prob += pulp.lpSum(
                        tasks[task['id'], t_start] * t_start + task['duration'] 
                    ) <= task['deadline']

        # целевая переменная
        prob += pulp.lpSum(
            tasks[task['id'], t_start] * task['reward'] for _, task in selected_tasks.iterrows() for t_start in range(max_time + 1)
        )

        prob.solve(pulp.PULP_CBC_CMD(msg=False))
        # self.show_profit(
        #     selected_tasks, start_time, y
        # )
        return pulp.value(prob.objective) if prob.status == 1 else 0

    def BnB_tree(self, depth, current_solution, current_reward, remaining_tasks):
        # current_solution содержит номера уже добавленых задач
        # в том порядке, в котором они будут в расписании
        # remaining_tasks - оставшиеся задачи для добавления

        # проверим корректность
        start_time = 0
        for i in current_solution:
            # задача должна укладываться в дедлайн
            if start_time + self.tasks.iloc[i]['duration'] > self.tasks.iloc[i]['deadline']:
                return False
            start_time += self.tasks.iloc[i]['duration']

        # оказались в листе
        if depth == self.n:
            if current_reward > self.best_reward:
                self.best_solution = current_solution.copy()
                self.best_reward = current_reward
            return
        
        ub = current_reward + sum(self.reward[remaining_tasks])

        # если верхняя граница меньше текущего рекорда
        if ub <= self.best_reward:
            return
                
        # если верхняя оценка ОСТАВШИХСЯ ЗАДАЧ не улучшает рекорд,
        # то не будем рассматривать решение дальше
        up_bound = current_reward + self.upper_bound_lp(self.tasks.iloc[remaining_tasks])
        if up_bound <= self.best_reward:
            return

        # из оставшися задач пытаемся добавить по очереди
        for i, task in enumerate(remaining_tasks):   
            # пытаемся добавить задачу
            next_remaining_tasks = np.concatenate((remaining_tasks[:i], remaining_tasks[i+1:]))
            # ветвимся влево
            left_branch = np.append(current_solution, task)
            left_reward = current_reward + self.tasks.iloc[task]['reward']
            
            self.BnB_tree(
                depth=depth+1,
                current_solution=left_branch,
                current_reward=left_reward,
                remaining_tasks=next_remaining_tasks
            )

            # пропускаеем задачу
            # ветвимся вправо
            # right_branch = np.append(current_solution, False)
            right_branch = current_solution
            right_reward = current_reward
            self.BnB_tree(
                depth=depth+1,
                current_solution=right_branch,
                current_reward=right_reward,
                remaining_tasks=next_remaining_tasks
            )


    
    def solve(self):
        self.BnB_tree(
            depth=0,
            current_solution=np.array([], dtype=int),
            current_reward=0,
            remaining_tasks=self.tasks.index.values
        )
        return self.best_solution, self.best_reward

def main():
    # input_filename = 'test_115.csv'
    # input_filename = 'test_142.csv'
    # input_filename = 'test_154.csv'
    # input_filename = 'test_84.csv'
    # input_filename = 'test_13.csv'
    input_filename = 'test_14.csv'
    
    # чтение из файла
    tasks, record = TaskScheduler.read_tasks_from_csv(input_filename)
    
    # Найдем оптимальное решение, используя алгоритм ветвей и границ и ЛП
    scheduler = TaskScheduler(tasks)
    optimal_schedule, total_reward = scheduler.solve()
    
    print(f"Total reward: {total_reward}")
    print(f"Optimal reward: {record}")
    print("Schedule:")
    assert total_reward == record
    start_time = 0
    for _, task in tasks.iloc[optimal_schedule].iterrows():
        print(f"Task {task['id']}: start at {start_time}, finish at {start_time + task['duration']} (reward: {task['reward']})")
        start_time += task['duration']

if __name__ == "__main__":
    main()
