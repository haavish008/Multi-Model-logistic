class QLearningRouteOptimizer:

    def __init__(
        self,
        graph,
        start,
        goal,
        learning_rate=0.10,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05
    ):

        self.graph = graph
        self.start = start
        self.goal = goal

        self.alpha = learning_rate
        self.gamma = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.q_table = {}

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def train(self, episodes=300, max_steps=None):

        if max_steps is None:
            max_steps = self.graph.number_of_nodes()

        history = []

        for episode in range(episodes):

            state = self.start
            visited = {state}
            total_reward = 0.0
            reached_goal = False

            for _ in range(max_steps):

                actions = list(self.graph.neighbors(state))

                if not actions:
                    break

                if random.random() < self.epsilon:
                    action = random.choice(actions)

                else:
                    action = max(
                        actions,
                        key=lambda a: self.get_q(state, a)
                    )

                cost = edge_cost(self.graph, state, action)

                reward = -cost / 1000.0

                if action in visited and action != self.goal:
                    reward -= 25.0

                if action == self.goal:
                    reward += 100.0
                    reached_goal = True

                next_actions = list(
                    self.graph.neighbors(action)
                )

                if action == self.goal:
                    future_reward = 0.0

                else:
                    future_reward = max(
                        (
                            self.get_q(action, next_action)
                            for next_action in next_actions
                        ),
                        default=0.0
                    )

                old_value = self.get_q(state, action)

                updated_value = old_value + self.alpha * (
                    reward
                    + self.gamma * future_reward
                    - old_value
                )

                self.q_table[(state, action)] = updated_value

                total_reward += reward
                state = action
                visited.add(state)

                if reached_goal:
                    break

            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.epsilon_decay
            )

            history.append({
                "episode": episode + 1,
                "reward": total_reward,
                "epsilon": self.epsilon,
                "reached_goal": reached_goal
            })

        return pd.DataFrame(history)

    def get_route(self):

        current = self.start
        route = [current]
        visited = {current}

        for _ in range(self.graph.number_of_nodes()):

            if current == self.goal:
                return route

            actions = [
                node
                for node in self.graph.neighbors(current)
                if node not in visited
            ]

            if not actions:
                break

            next_node = max(
                actions,
                key=lambda node: self.get_q(current, node)
            )

            if self.get_q(current, next_node) == 0:
                break

            route.append(next_node)
            visited.add(next_node)
            current = next_node

        # Fallback is important: an untrained policy may not reach the goal.
        return nx.astar_path(
            self.graph,
            self.start,
            self.goal,
            weight="dynamic_cost"
        )


