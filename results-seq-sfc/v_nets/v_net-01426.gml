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
  id 1426
  arrival_time 29935.22949912929
  lifetime 290.48070030176575
  num_nodes 11
  type "path"
  node [
    id 0
    label "0"
    cpu 30
    gpu 0
    rom 26
  ]
  node [
    id 1
    label "1"
    cpu 45
    gpu 32
    rom 47
  ]
  node [
    id 2
    label "2"
    cpu 16
    gpu 4
    rom 2
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 31
    rom 22
  ]
  node [
    id 4
    label "4"
    cpu 24
    gpu 6
    rom 6
  ]
  node [
    id 5
    label "5"
    cpu 5
    gpu 22
    rom 4
  ]
  node [
    id 6
    label "6"
    cpu 30
    gpu 0
    rom 18
  ]
  node [
    id 7
    label "7"
    cpu 49
    gpu 26
    rom 24
  ]
  node [
    id 8
    label "8"
    cpu 28
    gpu 4
    rom 43
  ]
  node [
    id 9
    label "9"
    cpu 35
    gpu 22
    rom 0
  ]
  node [
    id 10
    label "10"
    cpu 8
    gpu 33
    rom 47
  ]
  edge [
    source 0
    target 1
    bw 48
  ]
  edge [
    source 1
    target 2
    bw 8
  ]
  edge [
    source 2
    target 3
    bw 1
  ]
  edge [
    source 3
    target 4
    bw 47
  ]
  edge [
    source 4
    target 5
    bw 14
  ]
  edge [
    source 5
    target 6
    bw 29
  ]
  edge [
    source 6
    target 7
    bw 43
  ]
  edge [
    source 7
    target 8
    bw 18
  ]
  edge [
    source 8
    target 9
    bw 24
  ]
  edge [
    source 9
    target 10
    bw 43
  ]
]
