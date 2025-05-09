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
  id 694
  arrival_time 14633.028070873583
  lifetime 22.400863836856256
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 36
    gpu 44
    rom 5
  ]
  node [
    id 1
    label "1"
    cpu 44
    gpu 39
    rom 33
  ]
  node [
    id 2
    label "2"
    cpu 39
    gpu 31
    rom 29
  ]
  node [
    id 3
    label "3"
    cpu 28
    gpu 25
    rom 24
  ]
  node [
    id 4
    label "4"
    cpu 9
    gpu 48
    rom 47
  ]
  node [
    id 5
    label "5"
    cpu 32
    gpu 37
    rom 44
  ]
  node [
    id 6
    label "6"
    cpu 19
    gpu 46
    rom 7
  ]
  node [
    id 7
    label "7"
    cpu 26
    gpu 1
    rom 20
  ]
  node [
    id 8
    label "8"
    cpu 48
    gpu 0
    rom 46
  ]
  node [
    id 9
    label "9"
    cpu 45
    gpu 41
    rom 46
  ]
  edge [
    source 0
    target 1
    bw 33
  ]
  edge [
    source 1
    target 2
    bw 4
  ]
  edge [
    source 2
    target 3
    bw 6
  ]
  edge [
    source 3
    target 4
    bw 31
  ]
  edge [
    source 4
    target 5
    bw 46
  ]
  edge [
    source 5
    target 6
    bw 32
  ]
  edge [
    source 6
    target 7
    bw 46
  ]
  edge [
    source 7
    target 8
    bw 22
  ]
  edge [
    source 8
    target 9
    bw 46
  ]
]
