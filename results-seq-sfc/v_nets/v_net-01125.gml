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
  id 1125
  arrival_time 23485.205543300504
  lifetime 755.9908380721843
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 18
    gpu 20
    rom 2
  ]
  node [
    id 1
    label "1"
    cpu 16
    gpu 12
    rom 12
  ]
  node [
    id 2
    label "2"
    cpu 19
    gpu 21
    rom 3
  ]
  node [
    id 3
    label "3"
    cpu 46
    gpu 34
    rom 12
  ]
  node [
    id 4
    label "4"
    cpu 17
    gpu 16
    rom 47
  ]
  node [
    id 5
    label "5"
    cpu 50
    gpu 33
    rom 42
  ]
  node [
    id 6
    label "6"
    cpu 9
    gpu 44
    rom 14
  ]
  node [
    id 7
    label "7"
    cpu 21
    gpu 2
    rom 9
  ]
  node [
    id 8
    label "8"
    cpu 33
    gpu 29
    rom 39
  ]
  edge [
    source 0
    target 1
    bw 35
  ]
  edge [
    source 1
    target 2
    bw 26
  ]
  edge [
    source 2
    target 3
    bw 32
  ]
  edge [
    source 3
    target 4
    bw 44
  ]
  edge [
    source 4
    target 5
    bw 41
  ]
  edge [
    source 5
    target 6
    bw 37
  ]
  edge [
    source 6
    target 7
    bw 2
  ]
  edge [
    source 7
    target 8
    bw 29
  ]
]
