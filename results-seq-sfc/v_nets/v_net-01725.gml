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
  id 1725
  arrival_time 38535.48684932667
  lifetime 694.5428102180333
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 9
    gpu 15
    rom 2
  ]
  node [
    id 1
    label "1"
    cpu 42
    gpu 43
    rom 36
  ]
  node [
    id 2
    label "2"
    cpu 40
    gpu 0
    rom 28
  ]
  node [
    id 3
    label "3"
    cpu 4
    gpu 24
    rom 17
  ]
  node [
    id 4
    label "4"
    cpu 7
    gpu 10
    rom 27
  ]
  node [
    id 5
    label "5"
    cpu 28
    gpu 41
    rom 18
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 12
  ]
  edge [
    source 2
    target 3
    bw 29
  ]
  edge [
    source 3
    target 4
    bw 33
  ]
  edge [
    source 4
    target 5
    bw 6
  ]
]
