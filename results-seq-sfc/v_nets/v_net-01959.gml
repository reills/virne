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
  id 1959
  arrival_time 42914.99137804621
  lifetime 1376.68108067366
  num_nodes 12
  type "path"
  node [
    id 0
    label "0"
    cpu 27
    gpu 11
    rom 13
  ]
  node [
    id 1
    label "1"
    cpu 28
    gpu 8
    rom 0
  ]
  node [
    id 2
    label "2"
    cpu 25
    gpu 16
    rom 11
  ]
  node [
    id 3
    label "3"
    cpu 18
    gpu 46
    rom 27
  ]
  node [
    id 4
    label "4"
    cpu 7
    gpu 12
    rom 36
  ]
  node [
    id 5
    label "5"
    cpu 3
    gpu 5
    rom 48
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 34
    rom 0
  ]
  node [
    id 7
    label "7"
    cpu 47
    gpu 42
    rom 4
  ]
  node [
    id 8
    label "8"
    cpu 2
    gpu 38
    rom 23
  ]
  node [
    id 9
    label "9"
    cpu 9
    gpu 35
    rom 43
  ]
  node [
    id 10
    label "10"
    cpu 6
    gpu 10
    rom 27
  ]
  node [
    id 11
    label "11"
    cpu 14
    gpu 1
    rom 28
  ]
  edge [
    source 0
    target 1
    bw 36
  ]
  edge [
    source 1
    target 2
    bw 42
  ]
  edge [
    source 2
    target 3
    bw 2
  ]
  edge [
    source 3
    target 4
    bw 14
  ]
  edge [
    source 4
    target 5
    bw 18
  ]
  edge [
    source 5
    target 6
    bw 10
  ]
  edge [
    source 6
    target 7
    bw 23
  ]
  edge [
    source 7
    target 8
    bw 34
  ]
  edge [
    source 8
    target 9
    bw 17
  ]
  edge [
    source 9
    target 10
    bw 38
  ]
  edge [
    source 10
    target 11
    bw 32
  ]
]
