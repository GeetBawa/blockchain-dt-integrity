"""Local Ethereum test chain (py-evm via eth-tester) + thin wrappers around the contracts."""
import json, time
from pathlib import Path
from web3 import Web3, EthereumTesterProvider
from eth_tester import EthereumTester, PyEVMBackend

BUILD = Path(__file__).resolve().parent.parent / "build"


def load(name):
    d = json.loads((BUILD / f"{name}.json").read_text())
    return d["abi"], d["bytecode"]


class Chain:
    def __init__(self):
        self.tester = EthereumTester(backend=PyEVMBackend())
        self.w3 = Web3(EthereumTesterProvider(self.tester))
        self.accounts = self.w3.eth.accounts          # 10 funded test accounts
        self.admin = self.accounts[0]
        self.gateway = self.accounts[1]
        self.attacker = self.accounts[2]

    def deploy(self, name):
        abi, bc = load(name)
        c = self.w3.eth.contract(abi=abi, bytecode=bc)
        h = c.constructor().transact({"from": self.admin})
        r = self.w3.eth.wait_for_transaction_receipt(h)
        return self.w3.eth.contract(address=r.contractAddress, abi=abi), r.gasUsed

    def send(self, fn, sender):
        """Send a transaction, return (receipt, wall_seconds)."""
        t0 = time.perf_counter()
        h = fn.transact({"from": sender})
        r = self.w3.eth.wait_for_transaction_receipt(h)
        return r, time.perf_counter() - t0

    def block_number(self):
        return self.w3.eth.block_number
