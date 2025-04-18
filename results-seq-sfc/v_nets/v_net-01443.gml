graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1443
  arrival_time 30282.22828299886
  lifetime 629.4059285417143
  num_nodes 3
  type "path"
  node [
    id 0
    label "0"
    cpu 38
    gpu 12
    rom 4
  ]
  node [
    id 1
    label "1"
    cpu 12
    gpu 19
    rom 21
  ]
  node [
    id 2
    label "2"
    cpu 21
    gpu 42
    rom 34
  ]
  edge [
    source 0
    target 1
    bw 6
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
]
