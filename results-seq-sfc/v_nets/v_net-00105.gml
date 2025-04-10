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
  id 105
  arrival_time 2014.3468788195637
  lifetime 338.8815666656669
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 38
    gpu 10
    rom 16
  ]
  node [
    id 1
    label "1"
    cpu 50
    gpu 1
    rom 35
  ]
  node [
    id 2
    label "2"
    cpu 29
    gpu 38
    rom 34
  ]
  node [
    id 3
    label "3"
    cpu 13
    gpu 34
    rom 0
  ]
  node [
    id 4
    label "4"
    cpu 2
    gpu 49
    rom 16
  ]
  node [
    id 5
    label "5"
    cpu 33
    gpu 42
    rom 34
  ]
  node [
    id 6
    label "6"
    cpu 40
    gpu 4
    rom 33
  ]
  node [
    id 7
    label "7"
    cpu 48
    gpu 41
    rom 17
  ]
  node [
    id 8
    label "8"
    cpu 47
    gpu 42
    rom 3
  ]
  node [
    id 9
    label "9"
    cpu 18
    gpu 46
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 44
  ]
  edge [
    source 1
    target 2
    bw 27
  ]
  edge [
    source 2
    target 3
    bw 21
  ]
  edge [
    source 3
    target 4
    bw 31
  ]
  edge [
    source 4
    target 5
    bw 29
  ]
  edge [
    source 5
    target 6
    bw 6
  ]
  edge [
    source 6
    target 7
    bw 16
  ]
  edge [
    source 7
    target 8
    bw 31
  ]
  edge [
    source 8
    target 9
    bw 46
  ]
]
