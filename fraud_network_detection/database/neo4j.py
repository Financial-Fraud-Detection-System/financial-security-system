import logging

from neo4j import GraphDatabase

logger = logging.getLogger(__name__)


class Neo4jConnection:
    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password
        self.connected = False
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(
                self.uri, auth=(self.user, self.password)
            )
            self.connected = True
            logger.info("Neo4j connection established.")
        except Exception as e:
            self.connected = False
            logger.error(f"Failed to connect to Neo4j: {e}")

    def close(self):
        if self.connected:
            self.driver.close()
            logger.info("Neo4j connection closed.")
        else:
            logger.warning("No active Neo4j connection to close.")

    def create_account_node(self, tx, acc_num, features):
        query = (
            "CREATE (a:Account {acc_num: $acc_num, amt_mean: $amt_mean, amt_std: $amt_std, "
            "trans_count: $trans_count, state: $state, zip: $zip, city_pop: $city_pop, "
            "unix_time_mean: $unix_time_mean, time_span: $time_span, ip_trans_freq: $ip_trans_freq, "
            "ip_address: $ip_address})"
        )
        cleaned_features = {
            "acc_num": str(acc_num),
            "amt_mean": float(features["amt_mean"]),
            "amt_std": float(features["amt_std"]),
            "trans_count": int(max(features["trans_count"], 0)),
            "state": int(features["state"]),
            "zip": int(features["zip"]),
            "city_pop": float(features["city_pop"]),
            "unix_time_mean": float(features["unix_time_mean"]),
            "time_span": float(features["time_span"]),
            "ip_trans_freq": float(features["ip_trans_freq"]),
            "ip_address": str(features["ip_address"]),
        }
        tx.run(query, **cleaned_features)

    def create_edge(self, tx, src_acc, dst_acc, edge_type):
        query = (
            "MATCH (a:Account {acc_num: $src_acc}), (b:Account {acc_num: $dst_acc}) "
            f"CREATE (a)-[:{edge_type}]->(b)"
        )
        tx.run(query, src_acc=str(src_acc), dst_acc=str(dst_acc))

    def create_fraud_ring(self, tx, ring_id, accounts):
        query = (
            "CREATE (r:FraudRing {ring_id: $ring_id}) "
            "WITH r "
            "UNWIND $accounts AS acc "
            "MATCH (a:Account {acc_num: acc.acc_num}) "
            "CREATE (a)-[:IN_FRAUD_RING]->(r)"
        )
        tx.run(query, ring_id=ring_id, accounts=accounts)
