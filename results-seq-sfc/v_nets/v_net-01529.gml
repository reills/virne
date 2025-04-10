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
  id 1529
  arrival_time 33825.082983552886
  lifetime 306.21544428068773
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 15
    gpu 19
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 26
    gpu 11
    rom 45
  ]
  node [
    id 2
    label "2"
    cpu 2
    gpu 10
    rom 49
  ]
  node [
    id 3
    label "3"
    cpu 41
    gpu 38
    rom 0
  ]
  node [
    id 4
    label "4"
    cpu 14
    gpu 19
    rom 48
  ]
  node [
    id 5
    label "5"
    cpu 30
    gpu 37
    rom 39
  ]
  node [
    id 6
    label "6"
    cpu 29
    gpu 20
    rom 9
  ]
  node [
    id 7
    label "7"
    cpu 42
    gpu 27
    rom 46
  ]
  node [
    id 8
    label "8"
    cpu 41
    gpu 29
    rom 39
  ]
  edge [
    source 0
    target 1
    bw 32
  ]
  edge [
    source 1
    target 2
    bw 47
  ]
  edge [
    source 2
    target 3
    bw 25
  ]
  edge [
    source 3
    target 4
    bw 44
  ]
  edge [
    source 4
    target 5
    bw 12
  ]
  edge [
    source 5
    target 6
    bw 39
  ]
  edge [
    source 6
    target 7
    bw 0
  ]
  edge [
    source 7
    target 8
    bw 7
  ]
]
