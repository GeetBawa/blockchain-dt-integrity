const solc = require('/tmp/claude-0/-home-claude/2869df49-c3ed-593e-a3fd-957fc0c4994f/scratchpad/sol/node_modules/solc');
const fs = require('fs');
const src = fs.readFileSync('/home/claude/twin_chain/contracts/TwinProvenanceRegistry.sol', 'utf8');
const input = {
  language: 'Solidity',
  sources: { 'TwinProvenanceRegistry.sol': { content: src } },
  settings: { optimizer: { enabled: true, runs: 200 }, evmVersion: 'cancun',
    outputSelection: { '*': { '*': ['abi', 'evm.bytecode.object'] } } }
};
const out = JSON.parse(solc.compile(JSON.stringify(input)));
let fatal = false;
(out.errors || []).forEach(e => { console.log(e.severity + ': ' + e.formattedMessage); if (e.severity === 'error') fatal = true; });
if (fatal) process.exit(1);
for (const [file, cs] of Object.entries(out.contracts)) {
  for (const [name, c] of Object.entries(cs)) {
    fs.writeFileSync('/home/claude/twin_chain/build/' + name + '.json',
      JSON.stringify({ abi: c.abi, bytecode: '0x' + c.evm.bytecode.object }));
    console.log('compiled', name, 'bytecode bytes:', c.evm.bytecode.object.length / 2);
  }
}
