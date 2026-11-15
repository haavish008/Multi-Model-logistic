import heapq
import itertools


class DStarLite:

    def __init__(
        self,
        graph,
        start,
        goal
    ):

        self.graph = graph
        self.start = start
        self.last = start
        self.goal = goal
        self.km = 0.0
        self.g = {}
        self.rhs = {}
        self.queue = []
        self.counter = itertools.count()

        self.g[self.goal] = float("inf")
        self.rhs[self.goal] = 0.0

        self._push(self.goal)

    def _value(self, mapping, node):

        return mapping.get(
            node,
            float("inf")
        )

    def _heuristic(self, first, second):

        return math.hypot(
            first[0] - second[0],
            first[1] - second[1]
        )

    def _cost(self, first, second):

        if not self.graph.has_edge(
            first,
            second
        ):

            return float("inf")

        return float(
            self.graph[first][second].get(
                "dynamic_cost",
                self.graph[first][second].get(
                    "length_m",
                    float("inf")
                )
            )
        )

    def _key(self, node):

        value = min(
            self._value(self.g, node),
            self._value(self.rhs, node)
        )

        return (
            value + self._heuristic(
                self.start,
                node
            ) + self.km,
            value
        )

    def _push(self, node):

        key = self._key(node)

        heapq.heappush(
            self.queue,
            (
                key[0],
                key[1],
                next(self.counter),
                node
            )
        )

    def _top_key(self):

        while self.queue:

            k1, k2, _, node = self.queue[0]

            current_key = self._key(node)

            if (
                k1,
                k2
            ) == current_key:

                return current_key

            heapq.heappop(
                self.queue
            )

        return (
            float("inf"),
            float("inf")
        )

    def _pop(self):

        while self.queue:

            k1, k2, _, node = heapq.heappop(
                self.queue
            )

            current_key = self._key(node)

            if (
                k1,
                k2
            ) == current_key:

                return (
                    (k1, k2),
                    node
                )

        return (
            (float("inf"), float("inf")),
            None
        )

    def update_vertex(self, node):

        if node != self.goal:

            best = float("inf")

            for successor in self.graph.neighbors(
                node
            ):

                candidate = (
                    self._cost(
                        node,
                        successor
                    ) +
                    self._value(
                        self.g,
                        successor
                    )
                )

                best = min(
                    best,
                    candidate
                )

            self.rhs[node] = best

        if self._value(
            self.g,
            node
        ) != self._value(
            self.rhs,
            node
        ):

            self._push(
                node
            )

    def compute_shortest_path(self):

        iterations = 0

        maximum_iterations = max(
            10000,
            self.graph.number_of_nodes() * 100
        )

        while (
            self._top_key() < self._key(self.start)
            or
            self._value(
                self.rhs,
                self.start
            ) != self._value(
                self.g,
                self.start
            )
        ):

            iterations += 1

            if iterations > maximum_iterations:

                raise RuntimeError(
                    "D* Lite exceeded its iteration limit."
                )

            old_key, node = self._pop()

            if node is None:

                break

            new_key = self._key(
                node
            )

            if old_key < new_key:

                self._push(
                    node
                )

            elif self._value(
                self.g,
                node
            ) > self._value(
                self.rhs,
                node
            ):

                self.g[node] = self._value(
                    self.rhs,
                    node
                )

                for predecessor in self.graph.neighbors(
                    node
                ):

                    self.update_vertex(
                        predecessor
                    )

            else:

                self.g[node] = float("inf")

                self.update_vertex(
                    node
                )

                for predecessor in self.graph.neighbors(
                    node
                ):

                    self.update_vertex(
                        predecessor
                    )

    def move_start(self, new_start):

        self.km += self._heuristic(
            self.last,
            new_start
        )

        self.last = new_start
        self.start = new_start

    def update_edge_cost(
        self,
        first,
        second,
        new_cost
    ):

        if not self.graph.has_edge(
            first,
            second
        ):

            raise ValueError(
                "Cannot update an edge that does not exist."
            )

        self.graph[first][second][
            "dynamic_cost"
        ] = float(new_cost)

        self.update_vertex(
            first
        )

        self.update_vertex(
            second
        )

    def extract_path(self):

        self.compute_shortest_path()

        if not np.isfinite(
            self._value(
                self.g,
                self.start
            )
        ):

            raise nx.NetworkXNoPath(
                "D* Lite could not find a route."
            )

        current = self.start
        route = [current]
        visited = {
            current
        }

        maximum_steps = self.graph.number_of_nodes()

        for _ in range(
            maximum_steps
        ):

            if current == self.goal:

                return route

            candidates = []

            for successor in self.graph.neighbors(
                current
            ):

                score = (
                    self._cost(
                        current,
                        successor
                    ) +
                    self._value(
                        self.g,
                        successor
                    )
                )

                candidates.append(
                    (score, successor)
                )

            if not candidates:

                break

            _, next_node = min(
                candidates,
                key=lambda item: item[0]
            )

            if next_node in visited:

                raise RuntimeError(
                    "D* Lite path extraction encountered a loop."
                )

            route.append(
                next_node
            )

            visited.add(
                next_node
            )

            current = next_node

        raise nx.NetworkXNoPath(
            "D* Lite failed to reach the destination."
        )


def run_dstar_lite(
    graph,
    source,
    target
):

    logging.info(
        "Running D* Lite..."
    )

    planner = DStarLite(
        graph,
        source,
        target
    )

    route = planner.extract_path()

    logging.info(
        "D* Lite route generated with %d nodes.",
        len(route)
    )

    return planner, route


def simulate_road_blockage(
    planner,
    graph,
    first,
    second,
    penalty=1_000_000.0
):

    if not graph.has_edge(
        first,
        second
    ):

        raise ValueError(
            "Selected road segment does not exist."
        )

    logging.warning(
        "Simulating road blockage between %s and %s",
        first,
        second
    )

    planner.update_edge_cost(
        first,
        second,
        penalty
    )

    new_route = planner.extract_path()

    logging.info(
        "D* Lite recalculated the route."
    )

    return new_route


def export_dstar_results(
    graph,
    route,
    config,
    scenario_name="normal"
):

    rows = []

    for index, (u, v) in enumerate(
        zip(
            route[:-1],
            route[1:]
        ),
        start=1
    ):

        edge = graph[u][v]

        rows.append({
            "scenario": scenario_name,
            "sequence": index,
            "edge_cost": edge.get(
                "dynamic_cost",
                np.nan
            ),
            "distance_km": edge.get(
                "distance_km",
                edge["length_m"] / 1000.0
            ),
            "risk_score": edge.get(
                "risk_score",
                np.nan
            ),
            "geometry": edge["geometry"]
        })

    if not rows:

        return None

    result = gpd.GeoDataFrame(
        rows,
        geometry="geometry",
        crs=METRIC_CRS
    )

    output_name = (
        f"dstar_route_{scenario_name}.gpkg"
    )

    result.to_file(
        config.results_dir / output_name,
        layer="dstar_route",
        driver="GPKG"
    )

    return result


class RoadRoutingEnvironment:

    def __init__(
        self,
        graph,
        start,
        goal
    ):

        self.graph = graph
        self.start = start
        self.goal = goal
        self.current = start
        self.visited = set()
        self.steps = 0

        self.maximum_steps = max(
            100,
            graph.number_of_nodes()
        )

    def reset(self):

        self.current = self.start

        self.visited = {
            self.start
        }

        self.steps = 0

        return self.current

    def available_actions(self):

        return list(
            self.graph.neighbors(
                self.current
            )
        )

    def step(self, action):

        if not self.graph.has_edge(
            self.current,
            action
        ):

            return self.current, -1000.0, True

        edge = self.graph[
            self.current
        ][action]

        distance = edge.get(
            "distance_km",
            edge["length_m"] / 1000.0
        )

        risk = edge.get(
            "risk_score",
            0.0
        )

        cost = edge.get(
            "dynamic_cost",
            distance
        )

        reward = (
            -float(cost)
            - 10.0 * float(
                risk if np.isfinite(risk) else 0.0
            )
        )

        self.current = action
        self.steps += 1

        done = False

        if action == self.goal:

            reward += 100.0
            done = True

        elif action in self.visited:

            reward -= 50.0
            done = True

        elif self.steps >= self.maximum_steps:

            reward -= 50.0
            done = True

        self.visited.add(
            action
        )

        return (
            self.current,
            reward,
            done
        )


class QLearningAgent:

    def __init__(
        self,
        learning_rate=0.10,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.05,
        random_seed=42
    ):

        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.random = np.random.default_rng(
            random_seed
        )

        self.q_table = {}

    def get_q(self, state, action):

        return self.q_table.get(
            (state, action),
            0.0
        )

    def choose_action(
        self,
        state,
        actions
    ):

        if not actions:

            return None

        if self.random.random() < self.epsilon:

            return self.random.choice(
                actions
            )

        values = [
            self.get_q(
                state,
                action
            )
            for action in actions
        ]

        maximum = max(
            values
        )

        best_actions = [
            action
            for action, value in zip(
                actions,
                values
            )
            if value == maximum
        ]

        return self.random.choice(
            best_actions
        )

    def update(
        self,
        state,
        action,
        reward,
        next_state,
        next_actions,
        done
    ):

        old_value = self.get_q(
            state,
            action
        )

        if done or not next_actions:

            future_value = 0.0

        else:

            future_value = max(
                self.get_q(
                    next_state,
                    next_action
                )
                for next_action in next_actions
            )

        target = (
            reward +
            self.gamma * future_value
        )

        updated = (
            old_value +
            self.alpha * (
                target - old_value
            )
        )

        self.q_table[
            (state, action)
        ] = updated

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )


def train_q_learning(
    graph,
    source,
    target,
    episodes=500,
    learning_rate=0.10,
    discount_factor=0.95
):

    logging.info(
        "Training Q-learning routing agent..."
    )

    environment = RoadRoutingEnvironment(
        graph,
        source,
        target
    )

    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount_factor=discount_factor
    )

    history = []

    for episode in range(
        episodes
    ):

        state = environment.reset()
        total_reward = 0.0
        done = False

        while not done:

            actions = environment.available_actions()

            action = agent.choose_action(
                state,
                actions
            )

            if action is None:

                break

            next_state, reward, done = environment.step(
                action
            )

            next_actions = (
                environment.available_actions()
            )

            agent.update(
                state,
                action,
                reward,
                next_state,
                next_actions,
                done
            )

            state = next_state
            total_reward += reward

        agent.decay_epsilon()

        history.append({
            "episode": episode + 1,
            "total_reward": total_reward,
            "epsilon": agent.epsilon
        })

        if (episode + 1) % 50 == 0:

            logging.info(
                "Q-learning episode %d/%d | reward %.3f",
                episode + 1,
                episodes,
                total_reward
            )

    return agent, pd.DataFrame(
        history
    )


def extract_q_learning_route(
    graph,
    agent,
    source,
    target
):

    current = source
    route = [current]
    visited = {
        current
    }

    maximum_steps = graph.number_of_nodes()

    for _ in range(
        maximum_steps
    ):

        if current == target:

            return route

        actions = [
            node
            for node in graph.neighbors(
                current
            )
            if node not in visited
        ]

        if not actions:

            break

        next_node = max(
            actions,
            key=lambda node: agent.get_q(
                current,
                node
            )
        )

        if agent.get_q(
            current,
            next_node
        ) == 0.0:

            break

        route.append(
            next_node
        )

        visited.add(
            next_node
        )

        current = next_node

    logging.warning(
        "Q-learning did not produce a complete route. "
        "Using weighted shortest-path fallback."
    )

    return nx.shortest_path(
        graph,
        source,
        target,
        weight="dynamic_cost"
    )


def export_q_learning_results(
    history,
    route,
    graph,
    config
):

    history.to_csv(
        config.results_dir /
        "q_learning_training_history.csv",
        index=False
    )

    route_rows = []

    for index, (u, v) in enumerate(
        zip(
            route[:-1],
            route[1:]
        ),
        start=1
    ):

        edge = graph[u][v]

        route_rows.append({
            "sequence": index,
            "distance_km": edge.get(
                "distance_km",
                edge["length_m"] / 1000.0
            ),
            "dynamic_cost": edge.get(
                "dynamic_cost",
                np.nan
            ),
            "risk_score": edge.get(
                "risk_score",
                np.nan
            ),
            "geometry": edge["geometry"]
        })

    if route_rows:

        route_gdf = gpd.GeoDataFrame(
            route_rows,
            geometry="geometry",
            crs=METRIC_CRS
        )

        route_gdf.to_file(
            config.results_dir /
            "q_learning_route.gpkg",
            layer="q_learning_route",
            driver="GPKG"
        )

    return route_rows


def train_graph_neural_network(
    graph,
    config,
    epochs=100
):

    try:

        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        from torch_geometric.nn import GCNConv

    except ImportError:

        logging.warning(
            "PyTorch or PyTorch Geometric is unavailable. "
            "GNN training skipped."
        )

        logging.warning(
            "Install compatible torch and torch_geometric builds "
            "for your Python and operating system."
        )

        return None

    logging.info(
        "Preparing graph data for PyTorch Geometric..."
    )

    nodes = list(
        graph.nodes
    )

    node_to_index = {
        node: index
        for index, node in enumerate(nodes)
    }

    node_features = []
    node_labels = []

    for node in nodes:

        incident_edges = list(
            graph.edges(node, data=True)
        )

        risk_values = []
        rainfall_values = []
        elevation_values = []
        slope_values = []

        for _, _, edge in incident_edges:

            risk = edge.get(
                "risk_score",
                np.nan
            )

            rainfall = edge.get(
                "rainfall_mm",
                np.nan
            )

            elevation = edge.get(
                "elevation_m",
                np.nan
            )

            slope = edge.get(
                "slope_deg",
                np.nan
            )

            if np.isfinite(risk):

                risk_values.append(
                    risk
                )

            if np.isfinite(rainfall):

                rainfall_values.append(
                    rainfall
                )

            if np.isfinite(elevation):

                elevation_values.append(
                    elevation
                )

            if np.isfinite(slope):

                slope_values.append(
                    slope
                )

        mean_risk = (
            np.mean(risk_values)
            if risk_values
            else 0.0
        )

        mean_rainfall = (
            np.mean(rainfall_values)
            if rainfall_values
            else 0.0
        )

        mean_elevation = (
            np.mean(elevation_values)
            if elevation_values
            else 0.0
        )

        mean_slope = (
            np.mean(slope_values)
            if slope_values
            else 0.0
        )

        node_features.append([
            node[0] / 100000.0,
            node[1] / 100000.0,
            mean_rainfall / 100.0,
            mean_elevation / 3000.0,
            mean_slope / 90.0
        ])

        node_labels.append(
            mean_risk
        )

    edge_pairs = []

    for u, v in graph.edges:

        edge_pairs.append([
            node_to_index[u],
            node_to_index[v]
        ])

        edge_pairs.append([
            node_to_index[v],
            node_to_index[u]
        ])

    x = torch.tensor(
        node_features,
        dtype=torch.float
    )

    y = torch.tensor(
        node_labels,
        dtype=torch.float
    )

    edge_index = torch.tensor(
        edge_pairs,
        dtype=torch.long
    ).t().contiguous()

    data = Data(
        x=x,
        edge_index=edge_index,
        y=y
    )

    class RiskGCN(torch.nn.Module):

        def __init__(self, input_channels):

            super().__init__()

            self.conv1 = GCNConv(
                input_channels,
                32
            )

            self.conv2 = GCNConv(
                32,
                16
            )

            self.output = torch.nn.Linear(
                16,
                1
            )

        def forward(self, graph_data):

            features = self.conv1(
                graph_data.x,
                graph_data.edge_index
            )

            features = F.relu(
                features
            )

            features = F.dropout(
                features,
                p=0.2,
                training=self.training
            )

            features = self.conv2(
                features,
                graph_data.edge_index
            )

            features = F.relu(
                features
            )

            return torch.sigmoid(
                self.output(features).squeeze(-1)
            )

    model = RiskGCN(
        data.num_features
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.01,
        weight_decay=0.0001
    )

    loss_function = torch.nn.MSELoss()

    history = []

    model.train()

    for epoch in range(
        epochs
    ):

        optimizer.zero_grad()

        prediction = model(
            data
        )

        loss = loss_function(
            prediction,
            data.y
        )

        loss.backward()

        optimizer.step()

        history.append({
            "epoch": epoch + 1,
            "training_loss": float(
                loss.item()
            )
        })

        if (epoch + 1) % 20 == 0:

            logging.info(
                "GCN epoch %d/%d | loss %.6f",
                epoch + 1,
                epochs,
                loss.item()
            )

    model_path = (
        config.root /
        "models" /
        "uttarakhand_risk_gcn.pt"
    )

    model_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_features": data.num_features,
            "node_count": len(nodes),
            "feature_names": [
                "x_coordinate_scaled",
                "y_coordinate_scaled",
                "rainfall_scaled",
                "elevation_scaled",
                "slope_scaled"
            ]
        },
        model_path
    )

    model.eval()

    with torch.no_grad():

        predictions = model(
            data
        ).cpu().numpy()

    prediction_df = pd.DataFrame({
        "node_id": np.arange(
            len(nodes)
        ),
        "predicted_risk": predictions,
        "reference_risk": np.asarray(
            node_labels
        )
    })

    prediction_df.to_csv(
        config.results_dir /
        "gnn_node_risk_predictions.csv",
        index=False
    )

    pd.DataFrame(
        history
    ).to_csv(
        config.results_dir /
        "gnn_training_history.csv",
        index=False
    )

    logging.info(
        "GCN model saved: %s",
        model_path
    )

    return {
        "model": model,
        "predictions": prediction_df,
        "history": history
    }