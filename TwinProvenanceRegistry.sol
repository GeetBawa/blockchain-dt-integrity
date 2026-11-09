// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title Digital-twin provenance registry (FULL variant)
/// @notice Anchors Merkle roots of sensor-data batches, keeps a per-twin hash chain,
///         and records model lineage and twin decisions. Append-only by construction:
///         no function overwrites or deletes an existing anchor, model or decision.
contract TwinProvenanceRegistry {
    struct Anchor {
        bytes32 root;        // Merkle root of the batch
        address submitter;   // authorised gateway that anchored it
        uint64 timestamp;    // block time
        uint32 twinId;       // which twin (engine) the batch belongs to
    }

    struct Model {
        bytes32 modelHash;   // keccak256 of serialised model weights
        bytes32 dataHash;    // keccak256 of training data
        bytes32 configHash;  // keccak256 of hyper-parameters / pipeline config
        address registrant;
        uint64 timestamp;
    }

    struct Decision {
        bytes32 inputDigest;   // digest of the anchor roots the decision read
        bytes32 outputDigest;  // digest of the prediction
        uint64 timestamp;
        uint32 twinId;
        uint32 modelId;
    }

    address public admin;
    mapping(address => bool) public gateways;

    Anchor[] public anchors;
    mapping(uint32 => uint256[]) private twinAnchors;   // twinId -> anchor ids
    mapping(uint32 => bytes32) public twinHead;         // per-twin hash chain head

    Model[] public models;
    Decision[] public decisions;

    event Anchored(uint256 indexed id, uint32 indexed twinId, bytes32 root, uint32 startCycle, uint32 endCycle, bytes32 head);
    event ModelRegistered(uint256 indexed id, bytes32 modelHash, bytes32 dataHash, bytes32 configHash);
    event DecisionRecorded(uint256 indexed id, uint32 indexed twinId, uint32 modelId, bytes32 inputDigest, bytes32 outputDigest);

    error NotAdmin();
    error NotGateway();
    error BadRange();

    constructor() {
        admin = msg.sender;
        gateways[msg.sender] = true;
    }

    modifier onlyAdmin() { if (msg.sender != admin) revert NotAdmin(); _; }
    modifier onlyGateway() { if (!gateways[msg.sender]) revert NotGateway(); _; }

    function setGateway(address g, bool ok) external onlyAdmin { gateways[g] = ok; }

    function anchor(uint32 twinId, uint32 startCycle, uint32 endCycle, bytes32 root)
        external onlyGateway returns (uint256 id)
    {
        if (endCycle < startCycle) revert BadRange();
        id = anchors.length;
        anchors.push(Anchor(root, msg.sender, uint64(block.timestamp), twinId));
        twinAnchors[twinId].push(id);
        bytes32 head = keccak256(abi.encodePacked(twinHead[twinId], root));
        twinHead[twinId] = head;
        emit Anchored(id, twinId, root, startCycle, endCycle, head);
    }

    function registerModel(bytes32 modelHash, bytes32 dataHash, bytes32 configHash)
        external onlyGateway returns (uint256 id)
    {
        id = models.length;
        models.push(Model(modelHash, dataHash, configHash, msg.sender, uint64(block.timestamp)));
        emit ModelRegistered(id, modelHash, dataHash, configHash);
    }

    function recordDecision(uint32 twinId, uint32 modelId, bytes32 inputDigest, bytes32 outputDigest)
        external onlyGateway returns (uint256 id)
    {
        id = decisions.length;
        decisions.push(Decision(inputDigest, outputDigest, uint64(block.timestamp), twinId, modelId));
        emit DecisionRecorded(id, twinId, modelId, inputDigest, outputDigest);
    }

    function anchorCount() external view returns (uint256) { return anchors.length; }
    function twinAnchorIds(uint32 twinId) external view returns (uint256[] memory) { return twinAnchors[twinId]; }

    /// @notice Merkle inclusion proof (sorted-pair hashing, OpenZeppelin-compatible).
    function verifyProof(bytes32 root, bytes32 leaf, bytes32[] calldata proof) public pure returns (bool) {
        bytes32 h = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            bytes32 p = proof[i];
            h = h < p ? keccak256(abi.encodePacked(h, p)) : keccak256(abi.encodePacked(p, h));
        }
        return h == root;
    }

    function verifyAnchored(uint256 id, bytes32 leaf, bytes32[] calldata proof) external view returns (bool) {
        return verifyProof(anchors[id].root, leaf, proof);
    }
}

/// @title LEAN variant: one storage word per batch, everything else in events.
contract LeanRegistry {
    address public admin;
    mapping(address => bool) public gateways;
    mapping(bytes32 => bytes32) public roots;   // key = keccak(twinId, startCycle)

    event Anchored(uint32 indexed twinId, uint32 startCycle, uint32 endCycle, bytes32 root);

    error NotGateway();
    error AlreadyAnchored();

    constructor() { admin = msg.sender; gateways[msg.sender] = true; }

    function anchor(uint32 twinId, uint32 startCycle, uint32 endCycle, bytes32 root) external {
        if (!gateways[msg.sender]) revert NotGateway();
        bytes32 k = keccak256(abi.encodePacked(twinId, startCycle));
        if (roots[k] != bytes32(0)) revert AlreadyAnchored();
        roots[k] = root;
        emit Anchored(twinId, startCycle, endCycle, root);
    }
}
