import asyncio
import logging
from secrets import token_bytes
from typing import Any, Callable, Optional

import pytest
from blspy import G2Element

from chia.consensus.block_rewards import calculate_base_farmer_reward, calculate_pool_reward
from chia.full_node.mempool_manager import MempoolManager
from chia.rpc.wallet_rpc_api import WalletRpcApi
from chia.simulator.full_node_simulator import FullNodeSimulator
from chia.simulator.simulator_protocol import FarmNewBlockProtocol
from chia.simulator.time_out_assert import time_out_assert, time_out_assert_not_none
from chia.types.blockchain_format.program import Program
from chia.types.blockchain_format.sized_bytes import bytes32
from chia.types.coin_spend import CoinSpend
from chia.types.peer_info import PeerInfo
from chia.types.spend_bundle import SpendBundle
from chia.util.ints import uint16, uint32, uint64
from chia.wallet.did_wallet.did_wallet import DIDWallet
from chia.wallet.nft_wallet.nft_wallet import NFTWallet
from chia.wallet.outer_puzzles import create_asset_id, match_puzzle
from chia.wallet.puzzle_drivers import PuzzleInfo
from chia.wallet.puzzles.load_clvm import load_clvm
from chia.wallet.trading.offer import Offer
from chia.wallet.uncurried_puzzle import uncurry_puzzle
from chia.wallet.util.compute_memos import compute_memos
# from clvm_tools.binutils import disassemble
from chia.wallet.wallet import Wallet
from tests.util.wallet_is_synced import wallets_are_synced

logging.getLogger("aiosqlite").setLevel(logging.INFO)  # Too much logging on debug level

ROYALTY_SPLIT_MOD = load_clvm("royalty_split.clvm")

def mempool_not_empty(fnapi: FullNodeSimulator) -> bool:
    return len(fnapi.full_node.mempool_manager.mempool.spends) > 0


async def farm_blocks_until(predicate_f: Callable[[], bool], fnapi: FullNodeSimulator, ph: bytes32) -> None:
    for i in range(50):
        await fnapi.farm_new_transaction_block(FarmNewBlockProtocol(ph))
        if predicate_f():
            return None
        await asyncio.sleep(0.3)
    raise TimeoutError()


@pytest.mark.asyncio
# @pytest.mark.skip
async def test_nft_offer_sell_nft(two_wallet_nodes: Any) -> None:
    full_nodes, wallets, _ = two_wallet_nodes
    full_node_api: FullNodeSimulator = full_nodes[0]
    full_node_server = full_node_api.server
    wallet_node_maker, server_0 = wallets[0]
    wallet_node_taker, server_1 = wallets[1]
    api_0 = WalletRpcApi(wallet_node_maker)
    wallet_maker: Wallet = wallet_node_maker.wallet_state_manager.main_wallet
    wallet_taker = wallet_node_taker.wallet_state_manager.main_wallet

    ph_maker = await wallet_maker.get_new_puzzlehash()
    ph_taker = await wallet_taker.get_new_puzzlehash()
    ph_token = bytes32(token_bytes())

    wallet_node_maker.config["trusted_peers"] = {}
    wallet_node_taker.config["trusted_peers"] = {}

    await server_0.start_client(PeerInfo("localhost", uint16(full_node_server._port)), None)
    await server_1.start_client(PeerInfo("localhost", uint16(full_node_server._port)), None)

    await full_node_api.farm_new_transaction_block(FarmNewBlockProtocol(ph_maker))
    await full_node_api.farm_new_transaction_block(FarmNewBlockProtocol(ph_taker))
    await full_node_api.farm_new_transaction_block(FarmNewBlockProtocol(ph_token))
    await time_out_assert(20, wallets_are_synced, True, [wallet_node_maker, wallet_node_taker], full_node_api)

    funds = sum([calculate_pool_reward(uint32(i)) + calculate_base_farmer_reward(uint32(i)) for i in range(1, 2)])

    await time_out_assert(20, wallet_maker.get_unconfirmed_balance, funds)
    await time_out_assert(20, wallet_maker.get_confirmed_balance, funds)
    await time_out_assert(20, wallet_taker.get_unconfirmed_balance, funds)
    await time_out_assert(20, wallet_taker.get_confirmed_balance, funds)

    did_wallet_maker: DIDWallet = await DIDWallet.create_new_did_wallet(
        wallet_node_maker.wallet_state_manager, wallet_maker, uint64(1)
    )
    spend_bundle_list = await wallet_node_maker.wallet_state_manager.tx_store.get_unconfirmed_for_wallet(
        did_wallet_maker.id()
    )

    spend_bundle = spend_bundle_list[0].spend_bundle
    await time_out_assert_not_none(20, full_node_api.full_node.mempool_manager.get_spendbundle, spend_bundle.name())

    await full_node_api.farm_new_transaction_block(FarmNewBlockProtocol(ph_token))
    await time_out_assert(20, wallets_are_synced, True, [wallet_node_maker, wallet_node_taker], full_node_api)

    await time_out_assert(20, wallet_maker.get_pending_change_balance, 0)
    await time_out_assert(20, wallet_maker.get_unconfirmed_balance, funds - 1)
    await time_out_assert(20, wallet_maker.get_confirmed_balance, funds - 1)

    hex_did_id = did_wallet_maker.get_my_DID()
    did_id = bytes32.fromhex(hex_did_id)
    target_puzhash = ph_maker

    payout_scheme = Program.to([[ph_maker, 8000], [ph_taker, 2000]])
    royalty_split_puzzle = ROYALTY_SPLIT_MOD.curry(payout_scheme)
    royalty_puzhash = royalty_split_puzzle.get_tree_hash()
    royalty_basis_pts = uint16(200)

    nft_wallet_maker = await NFTWallet.create_new_nft_wallet(
        wallet_node_maker.wallet_state_manager, wallet_maker, name="NFT WALLET DID 1", did_id=did_id
    )
    metadata = Program.to(
        [
            ("u", ["https://www.chia.net/img/branding/chia-logo.svg"]),
            ("h", "0xD4584AD463139FA8C0D9F68F4B59F185"),
        ]
    )

    sb = await nft_wallet_maker.generate_new_nft(
        metadata,
        target_puzhash,
        royalty_puzhash,
        royalty_basis_pts,
        did_id,
    )
    assert sb
    # ensure hints are generated
    assert compute_memos(sb)
    await time_out_assert_not_none(20, full_node_api.full_node.mempool_manager.get_spendbundle, sb.name())

    await full_node_api.farm_new_transaction_block(FarmNewBlockProtocol(ph_token))
    await time_out_assert(20, wallets_are_synced, True, [wallet_node_maker, wallet_node_taker], full_node_api)

    await time_out_assert(20, len, 1, nft_wallet_maker.my_nft_coins)

    # TAKER SETUP -  NO DID
    nft_wallet_taker = await NFTWallet.create_new_nft_wallet(
        wallet_node_taker.wallet_state_manager, wallet_taker, name="NFT WALLET TAKER"
    )

    # maker create offer: NFT for xch
    trade_manager_maker = wallet_maker.wallet_state_manager.trade_manager
    trade_manager_taker = wallet_taker.wallet_state_manager.trade_manager

    coins_maker = nft_wallet_maker.my_nft_coins
    assert len(coins_maker) == 1
    coins_taker = nft_wallet_taker.my_nft_coins
    assert len(coins_taker) == 0

    nft_to_offer = coins_maker[0]
    nft_to_offer_info: Optional[PuzzleInfo] = match_puzzle(uncurry_puzzle(nft_to_offer.full_puzzle))
    nft_to_offer_asset_id: bytes32 = create_asset_id(nft_to_offer_info)  # type: ignore
    xch_requested = 1000
    maker_fee = uint64(433)

    offer_did_nft_for_xch = {nft_to_offer_asset_id: -1, wallet_maker.id(): xch_requested}

    success, trade_make, error = await trade_manager_maker.create_offer_for_ids(
        offer_did_nft_for_xch, {}, fee=maker_fee
    )

    assert success is True
    assert error is None
    assert trade_make is not None

    taker_fee = 1
    assert not mempool_not_empty(full_node_api)
    peer = wallet_node_taker.get_full_node_peer()
    assert peer is not None

    success, trade_take, error = await trade_manager_taker.respond_to_offer(
        Offer.from_bytes(trade_make.offer), peer, fee=uint64(taker_fee)
    )

    await time_out_assert(20, mempool_not_empty, True, full_node_api)

    assert error is None
    assert success is True
    assert trade_take is not None

    def maker_0_taker_1() -> bool:
        return len(nft_wallet_maker.my_nft_coins) == 0 and len(nft_wallet_taker.my_nft_coins) == 1

    await farm_blocks_until(maker_0_taker_1, full_node_api, ph_token)
    await time_out_assert(20, wallets_are_synced, True, [wallet_node_maker, wallet_node_taker], full_node_api)

    await time_out_assert(20, len, 0, nft_wallet_maker.my_nft_coins)
    await time_out_assert(20, len, 1, nft_wallet_taker.my_nft_coins)

    # assert payments and royalties before royalty split
    expected_royalty = uint64(xch_requested * royalty_basis_pts / 10000)
    expected_maker_balance = funds - 2 - maker_fee + xch_requested # no royalties have been received yet, since the split has not yet occurred
    expected_taker_balance = funds - taker_fee - xch_requested - expected_royalty
    await time_out_assert(20, wallet_maker.get_confirmed_balance, expected_maker_balance)
    await time_out_assert(20, wallet_taker.get_confirmed_balance, expected_taker_balance)

    royalty_coins = await full_node_api.full_node.coin_store.get_coin_records_by_puzzle_hash(False, royalty_puzhash)
    split_spend = SpendBundle([CoinSpend(royalty_coins[0].coin, royalty_split_puzzle, Program.to([royalty_coins[0].coin.amount]))], G2Element())

    await api_0.push_tx({"spend_bundle": bytes(split_spend).hex()})
    await time_out_assert_not_none(5, full_node_api.full_node.mempool_manager.get_spendbundle, split_spend.name())
    await full_node_api.farm_new_transaction_block(FarmNewBlockProtocol(ph_token))

    # assert payments and royalties after royalty split
    exected_royalty_maker = uint64(expected_royalty * 0.8)
    exected_royalty_taker = uint64(expected_royalty * 0.2)
    expected_maker_balance = funds - 2 - maker_fee + xch_requested + exected_royalty_maker
    expected_taker_balance = funds - taker_fee - xch_requested - expected_royalty + exected_royalty_taker
    await time_out_assert(20, wallet_maker.get_confirmed_balance, expected_maker_balance)
    await time_out_assert(20, wallet_taker.get_confirmed_balance, expected_taker_balance)
