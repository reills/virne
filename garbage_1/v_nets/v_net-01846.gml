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
  id 1846
  arrival_time 40802.010687155096
  lifetime 386.32245914012174
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 14
    rom 32
  ]
  node [
    id 1
    label "1"
    cpu 25
    gpu 13
    rom 31
  ]
  node [
    id 2
    label "2"
    cpu 9
    gpu 48
    rom 31
  ]
  node [
    id 3
    label "3"
    cpu 40
    gpu 50
    rom 33
  ]
  edge [
    source 0
    target 1
    bw 17
  ]
  edge [
    source 1
    target 2
    bw 42
  ]
  edge [
    source 2
    target 3
    bw 43
  ]
]
