import yaml

class TaskScheduler:
    def __init__(self, yaml_file):
        self.solver = ""
        self.tasks = []
        self.constraints = []
        self.objective = None
        self.load_yaml(yaml_file)

    def load_yaml(self, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            self.solver = data['solver']
            self.tasks = data['tasks']
            self.constraints = data['constraints']
            self.objective = data['objective']

    def build_solver(self) -> bool:
        if self.solver.lower() == 'pulp':
            try:
                import pulp
            except ModuleNotFoundError as ex:
                print(f"Решатель pulp не установлен")
                return False
        elif self.solver.lower() == 'cvxopt':
            try:
                import cvxopt
            except ModuleNotFoundError as ex:
                print(f"Решатель cvxopt не установлен")
                return False
        return True


    def build_pulp(self):
        # если не удалось подключить солвер
        if not self.build_solver():
            return
        import pulp

        prob = pulp.LpProblem("Task_Scheduling", pulp.LpMaximize)

        max_time = max([task['deadline'] for task in self.tasks])
        penalty_constant = max([task['duration'] for task in self.tasks]) + max_time
        # Переменные: время начала задач
        tasks = pulp.LpVariable.dicts(
            'task',
            ((task['id'], t_start) for task in self.tasks for t_start in range(max_time + 1)),
            cat='Binary'
        )

        # было ли опоздание
        is_late = pulp.LpVariable.dicts(
            'is_late',
            (task['id'] for task in self.tasks),
            lowBound=0,
            upBound=1,
            cat='Binary'
        )

         # каждая задача выполняется не более 1 раза
        for task in self.tasks:
            prob += pulp.lpSum(
                tasks[task['id'], t_start] for t_start in range(max_time + 1)
            ) <= 1

        # в каждый момент времени выполняется не более 1 задачи
        for t_start in range(max_time + 1):
            prob += pulp.lpSum(
                tasks[task['id'], t_start] for task in self.tasks
            ) <= 1
        
        # время старта + длительность <= дедлайна
        for task in self.tasks:
            M = 0
            # если есть штрафы, то не налагаем ограничение на дедлайн
            if 'penalty' in task:
                M = penalty_constant
            for t_start in range(max_time + 1):
                # если время выполнения меньше дедлайна
                if task['duration'] <= task['deadline']:
                    prob += pulp.lpSum(
                        tasks[task['id'], t_start] * t_start + task['duration'] 
                    ) <= task['deadline'] + M * is_late[task['id']]

        # Целевая функция: максимизация разницы прибыли и штрафов
        prob += (
            pulp.lpSum(
                tasks[task['id'], t_start] * task['reward'] for task in self.tasks for t_start in range(max_time + 1)
            ) - 
            pulp.lpSum(
                task['penalty'] * is_late[task['id']] for task in self.tasks if 'penalty' in task
            )
        )
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        ans = []
        for i, task in enumerate(self.tasks):
            for t_start in range(max_time + 1):
                if pulp.value(tasks[task['id'], t_start]) == 1:
                    ans.append(i)

        return ans

    def build_cvxopt(self):
        # если не удалось подключить солвер
        if not self.build_solver():
            return
        import cvxopt


    def solve(self):
        if self.solver == 'pulp':
            return self.build_pulp()
        elif self.solver == 'cvxopt':
            return self.build_cvxopt()


if __name__ == "__main__":
    scheduler = TaskScheduler("schedule.yaml")
    schedule = scheduler.solve()
    print("Оптимальное расписание:")
    start_time = 0
    for task_id in schedule:
        print(f"{scheduler.tasks[task_id]['id']}: start at {start_time}, finish at {start_time + scheduler.tasks[task_id]['duration']} "
              f"reward {scheduler.tasks[task_id]['reward']}")
        start_time += scheduler.tasks[task_id]['duration']