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
  id 785
  arrival_time 16189.362681188488
  lifetime 869.2996409008484
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 5
    gpu 11
    rom 22
  ]
  node [
    id 1
    label "1"
    cpu 5
    gpu 24
    rom 41
  ]
  node [
    id 2
    label "2"
    cpu 37
    gpu 50
    rom 12
  ]
  node [
    id 3
    label "3"
    cpu 3
    gpu 33
    rom 16
  ]
  edge [
    source 0
    target 1
    bw 19
  ]
  edge [
    source 1
    target 2
    bw 32
  ]
  edge [
    source 2
    target 3
    bw 24
  ]
]
