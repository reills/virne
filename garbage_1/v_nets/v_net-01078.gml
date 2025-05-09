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
  id 1078
  arrival_time 22562.259536743204
  lifetime 289.0214525977645
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 33
    gpu 1
    rom 27
  ]
  node [
    id 1
    label "1"
    cpu 47
    gpu 27
    rom 3
  ]
  node [
    id 2
    label "2"
    cpu 4
    gpu 37
    rom 2
  ]
  node [
    id 3
    label "3"
    cpu 12
    gpu 7
    rom 42
  ]
  edge [
    source 0
    target 1
    bw 48
  ]
  edge [
    source 1
    target 2
    bw 42
  ]
  edge [
    source 2
    target 3
    bw 9
  ]
]
