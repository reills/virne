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
  id 336
  arrival_time 6353.436968253491
  lifetime 1236.8879533846457
  num_nodes 6
  type "path"
  node [
    id 0
    label "0"
    cpu 37
    gpu 29
    rom 34
  ]
  node [
    id 1
    label "1"
    cpu 13
    gpu 29
    rom 44
  ]
  node [
    id 2
    label "2"
    cpu 30
    gpu 25
    rom 12
  ]
  node [
    id 3
    label "3"
    cpu 1
    gpu 5
    rom 11
  ]
  node [
    id 4
    label "4"
    cpu 41
    gpu 2
    rom 3
  ]
  node [
    id 5
    label "5"
    cpu 4
    gpu 45
    rom 25
  ]
  edge [
    source 0
    target 1
    bw 5
  ]
  edge [
    source 1
    target 2
    bw 40
  ]
  edge [
    source 2
    target 3
    bw 16
  ]
  edge [
    source 3
    target 4
    bw 20
  ]
  edge [
    source 4
    target 5
    bw 36
  ]
]
